/**
 * ZerodhaAutoLogin — TypeScript wrapper for the native Android Capacitor plugin.
 *
 * On Android (APK): calls the native ZerodhaAutoLoginPlugin Java class which
 *   - stores credentials in EncryptedSharedPreferences (Android Keystore)
 *   - schedules / cancels the daily 8 AM AlarmManager alarm
 *
 * On Web (browser): stores credentials in the backend DB via the REST API
 *   (since there is no native alarm system on the web, the Celery beat
 *   schedule acts as the fallback mechanism).
 *
 * Usage:
 *   import { ZerodhaAutoLogin } from './plugins/ZerodhaAutoLogin';
 *   await ZerodhaAutoLogin.saveCredentials({ userId, password, totpSecret, enabled });
 *   const status = await ZerodhaAutoLogin.getStatus();
 *   await ZerodhaAutoLogin.testLoginNow();
 */

import { registerPlugin, Capacitor } from '@capacitor/core';

export interface AutoLoginCredentials {
  userId: string;
  password: string;
  totpSecret: string;
  enabled: boolean;
}

export interface AutoLoginStatus {
  configured: boolean;
  enabled: boolean;
  userId: string;
}

export interface AutoLoginPlugin {
  saveCredentials(creds: AutoLoginCredentials): Promise<{ success: boolean; enabled: boolean; userId: string }>;
  getStatus(): Promise<AutoLoginStatus>;
  testLoginNow(): Promise<{ success: boolean; message: string }>;
}

// Register the native plugin — on Web this will be a no-op stub
const NativeAutoLogin = registerPlugin<AutoLoginPlugin>('ZerodhaAutoLogin');

/**
 * Platform-aware wrapper.
 *
 * Android  → delegates to the native plugin (real alarm + EncryptedSharedPrefs)
 * Web/iOS  → no-ops (the backend API handles persistence; Celery beat is the scheduler)
 */
export const ZerodhaAutoLogin = {
  isNative(): boolean {
    return Capacitor.isNativePlatform();
  },

  async saveCredentials(creds: AutoLoginCredentials): Promise<{ success: boolean; enabled: boolean; userId: string }> {
    if (Capacitor.isNativePlatform()) {
      return NativeAutoLogin.saveCredentials(creds);
    }
    // Web fallback: credentials are saved via the backend API in Settings.tsx
    return { success: true, enabled: creds.enabled, userId: creds.userId };
  },

  async getStatus(): Promise<AutoLoginStatus> {
    if (Capacitor.isNativePlatform()) {
      return NativeAutoLogin.getStatus();
    }
    // Web fallback: status comes from backend API
    return { configured: false, enabled: false, userId: '' };
  },

  async testLoginNow(): Promise<{ success: boolean; message: string }> {
    if (Capacitor.isNativePlatform()) {
      return NativeAutoLogin.testLoginNow();
    }
    return { success: false, message: 'Native auto-login only available on Android APK' };
  },
};
