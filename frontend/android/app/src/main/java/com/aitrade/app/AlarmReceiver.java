package com.aitrade.app;

import android.app.AlarmManager;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.util.Log;

import java.util.Calendar;

/**
 * AlarmReceiver — receives the daily 8:00 AM alarm and starts ZerodhaAutoLoginService.
 *
 * Also handles BOOT_COMPLETED so the alarm is re-registered after a phone reboot.
 *
 * Two broadcast actions handled:
 *   ACTION_BOOT_COMPLETED — re-schedule the alarm after reboot
 *   ACTION_ZERODHA_LOGIN  — the alarm fired; start the login service
 */
public class AlarmReceiver extends BroadcastReceiver {

    private static final String TAG = "AlarmReceiver";
    public static final String ACTION_ZERODHA_LOGIN = "com.aitrade.app.ZERODHA_AUTO_LOGIN";
    private static final int REQUEST_CODE = 8000;

    @Override
    public void onReceive(Context context, Intent intent) {
        String action = intent.getAction();
        Log.i(TAG, "onReceive: " + action);

        if (Intent.ACTION_BOOT_COMPLETED.equals(action)
                || Intent.ACTION_MY_PACKAGE_REPLACED.equals(action)) {
            // Re-register the alarm after reboot or app update
            scheduleDailyAlarm(context);

        } else if (ACTION_ZERODHA_LOGIN.equals(action)) {
            // Alarm fired — start the foreground login service
            Log.i(TAG, "8:00 AM alarm received — starting Zerodha auto-login");
            Intent serviceIntent = new Intent(context, ZerodhaAutoLoginService.class);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(serviceIntent);
            } else {
                context.startService(serviceIntent);
            }
        }
    }

    /**
     * Schedule (or re-schedule) the daily 8:00 AM IST exact alarm.
     * Must be called:
     *   - On first app launch / when user enables auto-login
     *   - After phone reboot (via BOOT_COMPLETED)
     *   - After each alarm fires (re-arm for next day)
     */
    public static void scheduleDailyAlarm(Context context) {
        AlarmManager am = (AlarmManager) context.getSystemService(Context.ALARM_SERVICE);
        if (am == null) return;

        Intent intent = new Intent(context, AlarmReceiver.class);
        intent.setAction(ACTION_ZERODHA_LOGIN);
        PendingIntent pi = PendingIntent.getBroadcast(
                context,
                REQUEST_CODE,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );

        // Set alarm for 8:00 AM local time (IST on device)
        Calendar cal = Calendar.getInstance();
        cal.set(Calendar.HOUR_OF_DAY, 8);
        cal.set(Calendar.MINUTE, 0);
        cal.set(Calendar.SECOND, 0);
        cal.set(Calendar.MILLISECOND, 0);

        // If 8 AM has already passed today, schedule for tomorrow
        if (cal.getTimeInMillis() <= System.currentTimeMillis()) {
            cal.add(Calendar.DAY_OF_YEAR, 1);
        }

        long triggerAt = cal.getTimeInMillis();
        Log.i(TAG, "Scheduling daily auto-login alarm for: " + cal.getTime());

        // Use setExactAndAllowWhileIdle for Doze-mode compatibility
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            am.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAt, pi);
        } else {
            am.setExact(AlarmManager.RTC_WAKEUP, triggerAt, pi);
        }
    }

    /**
     * Cancel the scheduled daily alarm (when user disables auto-login).
     */
    public static void cancelDailyAlarm(Context context) {
        AlarmManager am = (AlarmManager) context.getSystemService(Context.ALARM_SERVICE);
        if (am == null) return;

        Intent intent = new Intent(context, AlarmReceiver.class);
        intent.setAction(ACTION_ZERODHA_LOGIN);
        PendingIntent pi = PendingIntent.getBroadcast(
                context,
                REQUEST_CODE,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );
        am.cancel(pi);
        Log.i(TAG, "Daily auto-login alarm cancelled");
    }
}
