package br.com.murilovieira.fraudapi.service;

import br.com.murilovieira.fraudapi.domain.FraudResponse;
import br.com.murilovieira.fraudapi.domain.TransactionRequest;
import br.com.murilovieira.fraudapi.vector.VectorStore;
import br.com.murilovieira.fraudapi.vector.Vectorizer;
import org.springframework.stereotype.Service;

@Service
public class FraudService {

    private final Vectorizer  vectorizer;
    private final VectorStore vectorStore;

    public FraudService(Vectorizer vectorizer, VectorStore vectorStore) {
        this.vectorizer  = vectorizer;
        this.vectorStore = vectorStore;
    }

    public FraudResponse evaluate(TransactionRequest request) {
        float[] query  = VectorStore.threadQuery();
        vectorizer.vectorize(request, query);
        VectorStore.FraudResult result = vectorStore.search(query);
        return new FraudResponse(result.approved(), result.fraudScore());
    }
}
