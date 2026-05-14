package com.aitrade.app;

import android.annotation.SuppressLint;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.PixelFormat;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;
import android.view.WindowManager;
import android.webkit.CookieManager;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import androidx.core.app.NotificationCompat;
import androidx.security.crypto.EncryptedSharedPreferences;
import androidx.security.crypto.MasterKey;

import org.json.JSONObject;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * ZerodhaAutoLoginService — Foreground service that automates the daily
 * Zerodha OAuth login at 8:00 AM without any user interaction.
 *
 * Flow:
 *  1. Started by AlarmReceiver at 8:00 AM
 *  2. Shows a persistent foreground notification ("Logging in to Zerodha...")
 *  3. Creates a headless WebView and loads the Kite login URL
 *  4. JS injection fills: User ID → Next → Password → TOTP → Submit
 *  5. Intercepts the redirect URL → extracts request_token
 *  6. POSTs to /api/auth/callback → access_token saved on server
 *  7. POSTs to /api/auth/trigger-morning-trade → trading begins
 *  8. Updates notification: "✅ Connected to Zerodha" or "⚠️ Login failed"
 *  9. Stops itself after 90 seconds max
 */
public class ZerodhaAutoLoginService extends Service {

    private static final String TAG = "ZerodhaAutoLogin";
    public static final String CHANNEL_ID = "zerodha_autologin";
    public static final int NOTIF_ID = 8001;

    // Server base URL — update if your domain changes
    private static final String SERVER_BASE_URL = "http://nueroalgo.in";
    private static final int TIMEOUT_MS = 90_000; // 90 seconds max

    private WebView webView;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    // State tracking for multi-step login
    private enum LoginState { LOADING_LOGIN_PAGE, WAITING_USER_ID, WAITING_PASSWORD, WAITING_TOTP, DONE }
    private LoginState loginState = LoginState.LOADING_LOGIN_PAGE;
    private String requestToken = null;
    private String userId, password, totpSecret;

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.i(TAG, "ZerodhaAutoLoginService started");

        // Start as foreground service (required Android 8+)
        startForeground(NOTIF_ID, buildNotification("🔐 Logging in to Zerodha...", false));

        // Load credentials from EncryptedSharedPreferences
        if (!loadCredentials()) {
            Log.w(TAG, "No auto-login credentials found — stopping");
            updateNotification("⚠️ Auto-login not configured. Open app → Settings to set up.", true);
            stopSelfDelayed(5000);
            return START_NOT_STICKY;
        }

        // Check enabled flag
        try {
            MasterKey masterKey = new MasterKey.Builder(this)
                    .setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build();
            var prefs = EncryptedSharedPreferences.create(
                    this, "zerodha_auto_login",
                    masterKey,
                    EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                    EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            );
            boolean enabled = prefs.getBoolean("enabled", false);
            if (!enabled) {
                Log.i(TAG, "Auto-login disabled — skipping");
                stopSelf();
                return START_NOT_STICKY;
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to check enabled flag", e);
        }

        // Schedule hard timeout
        mainHandler.postDelayed(() -> {
            if (requestToken == null) {
                Log.e(TAG, "Login timed out after " + TIMEOUT_MS + "ms");
                updateNotification("⚠️ Zerodha auto-login timed out. Please login manually.", true);
                stopSelfDelayed(5000);
            }
        }, TIMEOUT_MS);

        // Start WebView login on main thread
        mainHandler.post(this::startWebViewLogin);

        return START_NOT_STICKY;
    }

    @SuppressLint("SetJavaScriptEnabled")
    private void startWebViewLogin() {
        // Build the Kite login URL
        String loginUrl = SERVER_BASE_URL + "/api/auth/login-url";
        executor.submit(() -> {
            try {
                URL url = new URL(loginUrl);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setConnectTimeout(10_000);
                conn.setReadTimeout(10_000);

                if (conn.getResponseCode() == 200) {
                    byte[] bytes = conn.getInputStream().readAllBytes();
                    JSONObject json = new JSONObject(new String(bytes, StandardCharsets.UTF_8));
                    String kiteLoginUrl = json.getString("login_url");
                    Log.i(TAG, "Got Kite login URL: " + kiteLoginUrl.substring(0, Math.min(60, kiteLoginUrl.length())));

                    mainHandler.post(() -> loadKiteUrl(kiteLoginUrl));
                } else {
                    Log.e(TAG, "Failed to get login URL: HTTP " + conn.getResponseCode());
                    updateNotification("⚠️ Could not reach server. Is it running?", true);
                    stopSelfDelayed(5000);
                }
            } catch (Exception e) {
                Log.e(TAG, "Error fetching login URL", e);
                updateNotification("⚠️ Network error: " + e.getMessage(), true);
                stopSelfDelayed(5000);
            }
        });
    }

    @SuppressLint({"SetJavaScriptEnabled", "AddJavascriptInterface"})
    private void loadKiteUrl(String kiteLoginUrl) {
        webView = new WebView(getApplicationContext());
        WebSettings ws = webView.getSettings();
        ws.setJavaScriptEnabled(true);
        ws.setDomStorageEnabled(true);
        ws.setUserAgentString(
            "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 " +
            "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
        );
        CookieManager.getInstance().setAcceptCookie(true);
        CookieManager.getInstance().setAcceptThirdPartyCookies(webView, true);

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                String url = request.getUrl().toString();
                Log.d(TAG, "URL: " + url.substring(0, Math.min(80, url.length())));

                // Zerodha redirects to our callback URL with request_token
                // Pattern: nueroalgo.in/api/auth/callback?request_token=XXXXX
                if (url.contains("request_token=")) {
                    String token = extractParam(url, "request_token");
                    if (token != null && !token.isEmpty()) {
                        requestToken = token;
                        Log.i(TAG, "Got request_token (" + token.length() + " chars)");
                        onRequestTokenReceived(token);
                        return true;
                    }
                }
                return false;
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                Log.d(TAG, "Page loaded: " + url);
                // Small delay to let React-based Kite page hydrate
                mainHandler.postDelayed(() -> handlePageLoaded(view, url), 1500);
            }
        });

        webView.loadUrl(kiteLoginUrl);
    }

    private void handlePageLoaded(WebView view, String url) {
        // Detect which step of Zerodha's login we're on and inject credentials
        if (url.contains("kite.zerodha.com") || url.contains("connect.zerodha.com")) {
            if (loginState == LoginState.LOADING_LOGIN_PAGE || loginState == LoginState.WAITING_USER_ID) {
                // Step 1: Fill user ID
                loginState = LoginState.WAITING_USER_ID;
                Log.i(TAG, "Injecting user ID");
                updateNotification("🔐 Entering Zerodha credentials...", false);

                // Zerodha Kite login page uses a React form
                String js = "javascript:(function() {" +
                    "var inputs = document.querySelectorAll('input[type=text], input[id*=user], input[name*=user], input[placeholder*=user], input[placeholder*=User]');" +
                    "for (var i = 0; i < inputs.length; i++) {" +
                    "  var el = inputs[i];" +
                    "  var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;" +
                    "  nativeInputValueSetter.call(el, '" + escapeJs(userId) + "');" +
                    "  el.dispatchEvent(new Event('input', { bubbles: true }));" +
                    "  el.dispatchEvent(new Event('change', { bubbles: true }));" +
                    "}" +
                    "})()";
                view.evaluateJavascript(js, null);

                // After filling user ID, submit to go to password step
                mainHandler.postDelayed(() -> {
                    String submitJs = "javascript:(function() {" +
                        "var btn = document.querySelector('button[type=submit]');" +
                        "if (btn) btn.click();" +
                        "})()";
                    view.evaluateJavascript(submitJs, null);
                    loginState = LoginState.WAITING_PASSWORD;
                }, 1000);

            } else if (loginState == LoginState.WAITING_PASSWORD) {
                // Step 2: Fill password
                Log.i(TAG, "Injecting password");
                String js = "javascript:(function() {" +
                    "var pInputs = document.querySelectorAll('input[type=password]');" +
                    "for (var i = 0; i < pInputs.length; i++) {" +
                    "  var el = pInputs[i];" +
                    "  var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;" +
                    "  nativeInputValueSetter.call(el, '" + escapeJs(password) + "');" +
                    "  el.dispatchEvent(new Event('input', { bubbles: true }));" +
                    "  el.dispatchEvent(new Event('change', { bubbles: true }));" +
                    "}" +
                    "})()";
                view.evaluateJavascript(js, null);

                mainHandler.postDelayed(() -> {
                    String submitJs = "javascript:(function() {" +
                        "var btn = document.querySelector('button[type=submit]');" +
                        "if (btn) btn.click();" +
                        "})()";
                    view.evaluateJavascript(submitJs, null);
                    loginState = LoginState.WAITING_TOTP;
                }, 1000);

            } else if (loginState == LoginState.WAITING_TOTP) {
                // Step 3: Fill TOTP
                String totp = TotpUtil.generateTotp(totpSecret);
                if (totp == null) {
                    Log.e(TAG, "TOTP generation failed");
                    updateNotification("⚠️ TOTP generation failed. Check your secret key.", true);
                    stopSelfDelayed(5000);
                    return;
                }
                Log.i(TAG, "Injecting TOTP code");
                updateNotification("🔢 Submitting TOTP code...", false);

                String js = "javascript:(function() {" +
                    "var inputs = document.querySelectorAll('input[type=text], input[inputmode=numeric], input[placeholder*=OTP], input[placeholder*=TOTP], input[placeholder*=2FA]');" +
                    "var tInput = inputs.length > 0 ? inputs[0] : null;" +
                    "if (!tInput) { tInput = document.querySelector('input'); }" +
                    "if (tInput) {" +
                    "  var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;" +
                    "  nativeInputValueSetter.call(tInput, '" + totp + "');" +
                    "  tInput.dispatchEvent(new Event('input', { bubbles: true }));" +
                    "  tInput.dispatchEvent(new Event('change', { bubbles: true }));" +
                    "}" +
                    "})()";
                view.evaluateJavascript(js, null);

                mainHandler.postDelayed(() -> {
                    String submitJs = "javascript:(function() {" +
                        "var btn = document.querySelector('button[type=submit]');" +
                        "if (btn) btn.click();" +
                        "})()";
                    view.evaluateJavascript(submitJs, null);
                    loginState = LoginState.DONE;
                }, 1000);
            }
        }
    }

    private void onRequestTokenReceived(String token) {
        loginState = LoginState.DONE;
        updateNotification("⚡ Exchanging token with server...", false);

        executor.submit(() -> {
            try {
                // Step 1: Exchange request_token for access_token
                String callbackUrl = SERVER_BASE_URL + "/api/auth/callback?request_token=" + token;
                int callbackStatus = httpGet(callbackUrl);
                if (callbackStatus != 200) {
                    throw new Exception("Callback returned HTTP " + callbackStatus);
                }
                Log.i(TAG, "Token exchanged successfully");

                // Step 2: Trigger morning trade sequence
                String tradeUrl = SERVER_BASE_URL + "/api/auth/trigger-morning-trade";
                httpPost(tradeUrl, "{}");
                Log.i(TAG, "Morning trade sequence triggered");

                mainHandler.post(() -> {
                    updateNotification("✅ Zerodha connected! Trading active.", true);
                    stopSelfDelayed(8000);
                });

            } catch (Exception e) {
                Log.e(TAG, "Post-token steps failed", e);
                mainHandler.post(() -> {
                    updateNotification("⚠️ Auth failed: " + e.getMessage(), true);
                    stopSelfDelayed(5000);
                });
            }
        });
    }

    // ── HTTP helpers ──────────────────────────────────────────────────────

    private int httpGet(String urlStr) throws Exception {
        URL url = new URL(urlStr);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(15_000);
        conn.setReadTimeout(15_000);
        int code = conn.getResponseCode();
        conn.disconnect();
        return code;
    }

    private void httpPost(String urlStr, String jsonBody) throws Exception {
        URL url = new URL(urlStr);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        conn.setConnectTimeout(15_000);
        conn.setReadTimeout(15_000);
        try (OutputStream os = conn.getOutputStream()) {
            os.write(jsonBody.getBytes(StandardCharsets.UTF_8));
        }
        conn.getResponseCode(); // execute
        conn.disconnect();
    }

    // ── Credential loading ────────────────────────────────────────────────

    private boolean loadCredentials() {
        try {
            MasterKey masterKey = new MasterKey.Builder(this)
                    .setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build();
            var prefs = EncryptedSharedPreferences.create(
                    this, "zerodha_auto_login",
                    masterKey,
                    EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                    EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            );
            userId = prefs.getString("user_id", null);
            password = prefs.getString("password", null);
            totpSecret = prefs.getString("totp_secret", null);
            return userId != null && !userId.isEmpty()
                    && password != null && !password.isEmpty()
                    && totpSecret != null && !totpSecret.isEmpty();
        } catch (Exception e) {
            Log.e(TAG, "Failed to load credentials", e);
            return false;
        }
    }

    // ── Notification helpers ──────────────────────────────────────────────

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel ch = new NotificationChannel(
                    CHANNEL_ID, "Zerodha Auto-Login",
                    NotificationManager.IMPORTANCE_LOW
            );
            ch.setDescription("Daily automated Zerodha authentication at 8 AM");
            ((NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE)).createNotificationChannel(ch);
        }
    }

    private Notification buildNotification(String text, boolean done) {
        Intent intent = new Intent(this, MainActivity.class);
        PendingIntent pi = PendingIntent.getActivity(this, 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("AItrade — Auto-Login")
                .setContentText(text)
                .setSmallIcon(done ? android.R.drawable.stat_sys_download_done : android.R.drawable.stat_notify_sync)
                .setContentIntent(pi)
                .setOngoing(!done)
                .setAutoCancel(done)
                .build();
    }

    private void updateNotification(String text, boolean done) {
        NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        if (nm != null) nm.notify(NOTIF_ID, buildNotification(text, done));
    }

    private void stopSelfDelayed(long delayMs) {
        mainHandler.postDelayed(this::stopSelf, delayMs);
    }

    // ── Utilities ─────────────────────────────────────────────────────────

    private static String extractParam(String url, String paramName) {
        try {
            String search = paramName + "=";
            int start = url.indexOf(search);
            if (start < 0) return null;
            start += search.length();
            int end = url.indexOf("&", start);
            return end < 0 ? url.substring(start) : url.substring(start, end);
        } catch (Exception e) {
            return null;
        }
    }

    private static String escapeJs(String s) {
        return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "");
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (webView != null) {
            mainHandler.post(() -> {
                webView.stopLoading();
                webView.destroy();
                webView = null;
            });
        }
        executor.shutdownNow();
        Log.i(TAG, "ZerodhaAutoLoginService destroyed");
    }
}
