package br.com.murilovieira.fraudapi.config;

import br.com.murilovieira.fraudapi.domain.FraudResponse;
import br.com.murilovieira.fraudapi.domain.TransactionRequest;
import org.springframework.aot.hint.MemberCategory;
import org.springframework.aot.hint.RuntimeHints;
import org.springframework.aot.hint.RuntimeHintsRegistrar;

public class ApplicationHints implements RuntimeHintsRegistrar {

    @Override
    public void registerHints(RuntimeHints hints, ClassLoader classLoader) {
        hints.resources()
                .registerPattern("normalization.json")
                .registerPattern("mcc_risk.json");

        hints.reflection()
                .registerType(NormalizationConfig.class,                MemberCategory.INVOKE_DECLARED_CONSTRUCTORS)
                .registerType(TransactionRequest.class,                 MemberCategory.INVOKE_DECLARED_CONSTRUCTORS)
                .registerType(TransactionRequest.Transaction.class,     MemberCategory.INVOKE_DECLARED_CONSTRUCTORS)
                .registerType(TransactionRequest.Customer.class,        MemberCategory.INVOKE_DECLARED_CONSTRUCTORS)
                .registerType(TransactionRequest.Merchant.class,        MemberCategory.INVOKE_DECLARED_CONSTRUCTORS)
                .registerType(TransactionRequest.Terminal.class,        MemberCategory.INVOKE_DECLARED_CONSTRUCTORS)
                .registerType(TransactionRequest.LastTransaction.class, MemberCategory.INVOKE_DECLARED_CONSTRUCTORS)
                .registerType(FraudResponse.class,                      MemberCategory.INVOKE_DECLARED_CONSTRUCTORS);
    }
}
