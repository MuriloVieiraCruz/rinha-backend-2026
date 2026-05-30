package br.com.murilovieira.fraudapi.domain;

import com.fasterxml.jackson.annotation.JsonProperty;

public record FraudResponse(
        boolean approved,
        @JsonProperty("fraud_score") double fraudScore
) {}
