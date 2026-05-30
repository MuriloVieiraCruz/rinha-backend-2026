package br.com.murilovieira.fraudapi.vector;

import br.com.murilovieira.fraudapi.domain.TransactionRequest;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;
import tools.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Map;

@Component
public class Vectorizer {

    static final int DIM = 14;

    private static final float DEFAULT_MCC_RISK = 0.5f;

    private final double maxAmount;
    private final double maxInstallments;
    private final double amountVsAvgRatio;
    private final double maxMinutes;
    private final double maxKm;
    private final double maxTxCount24h;
    private final double maxMerchantAvgAmount;

    private final Map<String, Float> mccRiskMap;

    private final ThreadLocal<float[]> vectorPool =
            ThreadLocal.withInitial(() -> new float[DIM]);

    public Vectorizer(ObjectMapper objectMapper) throws IOException {
        var norm = objectMapper.readValue(
                new ClassPathResource("normalization.json").getInputStream(),
                NormalizationData.class
        );
        this.maxAmount            = norm.maxAmount();
        this.maxInstallments      = norm.maxInstallments();
        this.amountVsAvgRatio     = norm.amountVsAvgRatio();
        this.maxMinutes           = norm.maxMinutes();
        this.maxKm                = norm.maxKm();
        this.maxTxCount24h        = norm.maxTxCount24h();
        this.maxMerchantAvgAmount = norm.maxMerchantAvgAmount();

        var mapType = objectMapper.getTypeFactory()
                .constructMapType(Map.class, String.class, Float.class);
        this.mccRiskMap = objectMapper.readValue(
                new ClassPathResource("mcc_risk.json").getInputStream(),
                mapType
        );
    }

    public float[] vectorize(TransactionRequest req) {
        float[] vec = vectorPool.get();

        var tx       = req.transaction();
        var customer = req.customer();
        var merchant = req.merchant();
        var terminal = req.terminal();
        var lastTx   = req.lastTransaction();

        Instant txInstant = Instant.parse(tx.requestedAt());
        var     zdt       = txInstant.atOffset(java.time.ZoneOffset.UTC);

        vec[0]  = clamp(tx.amount() / maxAmount);
        vec[1]  = clamp(tx.installments() / maxInstallments);
        vec[2]  = customer.avgAmount() > 0
                    ? clamp((tx.amount() / customer.avgAmount()) / amountVsAvgRatio)
                    : 0f;
        vec[3]  = (float) (zdt.getHour() / 23.0);
        vec[4]  = (float) ((zdt.getDayOfWeek().getValue() - 1) / 6.0);

        if (lastTx != null) {
            Instant lastInstant   = Instant.parse(lastTx.timestamp());
            long    minutesDelta  = Duration.between(lastInstant, txInstant).toMinutes();
            vec[5] = clamp(minutesDelta / maxMinutes);
            vec[6] = clamp(lastTx.kmFromCurrent() / maxKm);
        } else {
            vec[5] = -1f;
            vec[6] = -1f;
        }

        vec[7]  = clamp(terminal.kmFromHome() / maxKm);
        vec[8]  = clamp(customer.txCount24h() / maxTxCount24h);
        vec[9]  = terminal.isOnline()    ? 1f : 0f;
        vec[10] = terminal.cardPresent() ? 1f : 0f;
        vec[11] = isKnownMerchant(merchant.id(), customer.knownMerchants()) ? 0f : 1f;
        vec[12] = mccRiskMap.getOrDefault(merchant.mcc(), DEFAULT_MCC_RISK);
        vec[13] = clamp(merchant.avgAmount() / maxMerchantAvgAmount);

        return vec;
    }

    private static boolean isKnownMerchant(String merchantId, String[] knownMerchants) {
        if (knownMerchants == null || merchantId == null) return false;
        for (String m : knownMerchants) {
            if (merchantId.equals(m)) return true;
        }
        return false;
    }

    private static float clamp(double value) {
        if (value < 0.0) return 0f;
        if (value > 1.0) return 1f;
        return (float) value;
    }

    private record NormalizationData(
            @JsonProperty("max_amount")              double maxAmount,
            @JsonProperty("max_installments")        double maxInstallments,
            @JsonProperty("amount_vs_avg_ratio")     double amountVsAvgRatio,
            @JsonProperty("max_minutes")             double maxMinutes,
            @JsonProperty("max_km")                  double maxKm,
            @JsonProperty("max_tx_count_24h")        double maxTxCount24h,
            @JsonProperty("max_merchant_avg_amount") double maxMerchantAvgAmount
    ) {}
}
