package com.aitrade.app;

import android.util.Base64;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.ByteBuffer;

/**
 * RFC 6238 TOTP (Time-based One-Time Password) implementation.
 *
 * Produces the same 6-digit code as Google Authenticator for a given
 * base32-encoded secret key.  No external library required — standard
 * Android HMAC-SHA1 + a minimal base32 decoder.
 *
 * Usage:
 *   String code = TotpUtil.generateTotp(totpSecretBase32);
 *   // Returns "123456" (always 6 digits, zero-padded)
 */
public class TotpUtil {

    private static final int TOTP_DIGITS = 6;
    private static final int TOTP_PERIOD = 30; // seconds
    private static final String HMAC_ALGO = "HmacSHA1";

    /**
     * Generate the current 6-digit TOTP code.
     *
     * @param base32Secret  The base32-encoded secret (as shown in Zerodha 2FA setup).
     *                      Spaces and hyphens are stripped automatically.
     * @return 6-digit TOTP string (e.g. "042817"), or null on failure.
     */
    public static String generateTotp(String base32Secret) {
        try {
            byte[] key = base32Decode(base32Secret);
            long timeStep = System.currentTimeMillis() / 1000L / TOTP_PERIOD;
            byte[] data = ByteBuffer.allocate(8).putLong(timeStep).array();

            Mac mac = Mac.getInstance(HMAC_ALGO);
            mac.init(new SecretKeySpec(key, HMAC_ALGO));
            byte[] hash = mac.doFinal(data);

            // Dynamic truncation (RFC 4226 §5.4)
            int offset = hash[hash.length - 1] & 0x0F;
            int binary = ((hash[offset] & 0x7F) << 24)
                    | ((hash[offset + 1] & 0xFF) << 16)
                    | ((hash[offset + 2] & 0xFF) << 8)
                    | (hash[offset + 3] & 0xFF);

            int otp = binary % (int) Math.pow(10, TOTP_DIGITS);
            return String.format("%0" + TOTP_DIGITS + "d", otp);

        } catch (Exception e) {
            android.util.Log.e("TotpUtil", "TOTP generation failed", e);
            return null;
        }
    }

    /**
     * Minimal base32 decoder (RFC 4648).
     * Handles uppercase and lowercase input; strips spaces and hyphens.
     */
    private static byte[] base32Decode(String input) {
        String cleaned = input.toUpperCase().replaceAll("[\\s\\-=]", "");
        final String ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
        int buffer = 0;
        int bitsLeft = 0;
        byte[] result = new byte[cleaned.length() * 5 / 8];
        int pos = 0;

        for (char c : cleaned.toCharArray()) {
            int val = ALPHABET.indexOf(c);
            if (val < 0) continue;
            buffer = (buffer << 5) | val;
            bitsLeft += 5;
            if (bitsLeft >= 8) {
                result[pos++] = (byte) ((buffer >> (bitsLeft - 8)) & 0xFF);
                bitsLeft -= 8;
            }
        }
        // Trim to actual size
        byte[] trimmed = new byte[pos];
        System.arraycopy(result, 0, trimmed, 0, pos);
        return trimmed;
    }
}
