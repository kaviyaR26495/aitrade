package com.aitrade.app;

import android.content.Context;
import android.util.Log;

import androidx.security.crypto.EncryptedSharedPreferences;
import androidx.security.crypto.MasterKey;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

/**
 * ZerodhaAutoLoginPlugin — Capacitor bridge between the React/TypeScript
 * frontend and the native Android auto-login system.
 *
 * Exposes three methods to JavaScript:
 *   saveCredentials(userId, password, totpSecret, enabled)
 *     → stores in EncryptedSharedPreferences + schedules/cancels alarm
 *   getStatus()
 *     → returns { configured, enabled, userId }
 *   testLoginNow()
 *     → fires the login service immediately for testing
 */
@CapacitorPlugin(name = "ZerodhaAutoLogin")
public class ZerodhaAutoLoginPlugin extends Plugin {

    private static final String TAG = "ZerodhaAutoLoginPlugin";
    private static final String PREFS_NAME = "zerodha_auto_login";

    @PluginMethod
    public void saveCredentials(PluginCall call) {
        String userId = call.getString("userId", "");
        String password = call.getString("password", "");
        String totpSecret = call.getString("totpSecret", "");
        boolean enabled = Boolean.TRUE.equals(call.getBoolean("enabled", false));

        if (userId.isEmpty() || password.isEmpty() || totpSecret.isEmpty()) {
            call.reject("userId, password, and totpSecret are required");
            return;
        }

        try {
            MasterKey masterKey = new MasterKey.Builder(getContext())
                    .setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build();

            var prefs = EncryptedSharedPreferences.create(
                    getContext(),
                    PREFS_NAME,
                    masterKey,
                    EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                    EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            );

            prefs.edit()
                    .putString("user_id", userId)
                    .putString("password", password)
                    .putString("totp_secret", totpSecret)
                    .putBoolean("enabled", enabled)
                    .apply();

            // Schedule or cancel alarm based on enabled flag
            if (enabled) {
                AlarmReceiver.scheduleDailyAlarm(getContext());
                Log.i(TAG, "Credentials saved + daily alarm scheduled for user: " + userId);
            } else {
                AlarmReceiver.cancelDailyAlarm(getContext());
                Log.i(TAG, "Credentials saved + alarm cancelled (disabled)");
            }

            JSObject result = new JSObject();
            result.put("success", true);
            result.put("enabled", enabled);
            result.put("userId", userId);
            call.resolve(result);

        } catch (Exception e) {
            Log.e(TAG, "Failed to save credentials", e);
            call.reject("Failed to save credentials: " + e.getMessage());
        }
    }

    @PluginMethod
    public void getStatus(PluginCall call) {
        try {
            MasterKey masterKey = new MasterKey.Builder(getContext())
                    .setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build();

            var prefs = EncryptedSharedPreferences.create(
                    getContext(),
                    PREFS_NAME,
                    masterKey,
                    EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                    EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            );

            String userId = prefs.getString("user_id", "");
            String password = prefs.getString("password", "");
            String totpSecret = prefs.getString("totp_secret", "");
            boolean enabled = prefs.getBoolean("enabled", false);
            boolean configured = !userId.isEmpty() && !password.isEmpty() && !totpSecret.isEmpty();

            JSObject result = new JSObject();
            result.put("configured", configured);
            result.put("enabled", enabled);
            result.put("userId", userId);
            call.resolve(result);

        } catch (Exception e) {
            Log.e(TAG, "Failed to get status", e);
            // Not configured — return defaults
            JSObject result = new JSObject();
            result.put("configured", false);
            result.put("enabled", false);
            result.put("userId", "");
            call.resolve(result);
        }
    }

    @PluginMethod
    public void testLoginNow(PluginCall call) {
        try {
            android.content.Intent intent = new android.content.Intent(
                    getContext(), ZerodhaAutoLoginService.class
            );
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                getContext().startForegroundService(intent);
            } else {
                getContext().startService(intent);
            }
            Log.i(TAG, "Test login triggered manually");
            JSObject result = new JSObject();
            result.put("success", true);
            result.put("message", "Login service started");
            call.resolve(result);
        } catch (Exception e) {
            Log.e(TAG, "Failed to start login service", e);
            call.reject("Failed to start: " + e.getMessage());
        }
    }
}
