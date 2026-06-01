package br.com.murilovieira.fraudapi.vector;

import br.com.murilovieira.fraudapi.config.NormalizationConfig;
import br.com.murilovieira.fraudapi.domain.TransactionRequest;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class Vectorizer {

    private final NormalizationConfig norm;
    private final Map<String, Float>  mccRisk;

    public Vectorizer(NormalizationConfig norm, Map<String, Float> mccRisk) {
        this.norm    = norm;
        this.mccRisk = mccRisk;
    }

    public void vectorize(TransactionRequest req, float[] v) {
        var tx       = req.transaction();
        var customer = req.customer();
        var merchant = req.merchant();
        var terminal = req.terminal();
        var lastTx   = req.lastTransaction();

        v[0] = clamp(tx.amount()       / norm.maxAmount());
        v[1] = clamp(tx.installments() / norm.maxInstallments());
        v[2] = customer.avgAmount() > 0
                ? clamp((tx.amount() / customer.avgAmount()) / norm.amountVsAvgRatio())
                : 0f;

        long epochSec  = parseIsoEpochSec(tx.requestedAt());
        long epochDays = Math.floorDiv(epochSec, 86400L);
        int  hour      = (int) (Math.floorMod(epochSec, 86400L) / 3600L);
        int  dow       = (int) Math.floorMod(epochDays + 3L, 7L);
        v[3] = hour / 23f;
        v[4] = dow  / 6f;

        if (lastTx != null) {
            long lastEpoch = parseIsoEpochSec(lastTx.timestamp());
            long minutes   = Math.abs(epochSec - lastEpoch) / 60L;
            v[5] = clamp(minutes / norm.maxMinutes());
            v[6] = clamp(lastTx.kmFromCurrent() / norm.maxKm());
        } else {
            v[5] = -1f;
            v[6] = -1f;
        }

        v[7]  = clamp(terminal.kmFromHome() / norm.maxKm());
        v[8]  = clamp(customer.txCount24h() / norm.maxTxCount24h());
        v[9]  = terminal.isOnline()    ? 1f : 0f;
        v[10] = terminal.cardPresent() ? 1f : 0f;
        v[11] = containsMerchant(customer.knownMerchants(), merchant.id()) ? 0f : 1f;
        v[12] = mccRisk.getOrDefault(merchant.mcc(), 0.5f);
        v[13] = clamp(merchant.avgAmount() / norm.maxMerchantAvgAmount());
    }

    private static long parseIsoEpochSec(String s) {
        int year   = digit4(s, 0);
        int month  = digit2(s, 5);
        int day    = digit2(s, 8);
        int hour   = digit2(s, 11);
        int minute = digit2(s, 14);
        int second = digit2(s, 17);
        return daysFromCivil(year, month, day) * 86400L + hour * 3600L + minute * 60L + second;
    }

    private static int digit2(String s, int i) {
        return (s.charAt(i) - '0') * 10 + (s.charAt(i + 1) - '0');
    }

    private static int digit4(String s, int i) {
        return (s.charAt(i)     - '0') * 1000
             + (s.charAt(i + 1) - '0') * 100
             + (s.charAt(i + 2) - '0') * 10
             + (s.charAt(i + 3) - '0');
    }

    private static long daysFromCivil(int y, int m, int d) {
        y -= m <= 2 ? 1 : 0;
        int era = (y >= 0 ? y : y - 399) / 400;
        int yoe = y - era * 400;
        int doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
        int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        return (long) era * 146097L + doe - 719468L;
    }

    private static float clamp(double x) {
        return (float) Math.min(1.0, Math.max(0.0, x));
    }

    private static boolean containsMerchant(String[] arr, String target) {
        if (arr == null || target == null) return false;
        for (String m : arr) if (target.equals(m)) return true;
        return false;
    }
}
