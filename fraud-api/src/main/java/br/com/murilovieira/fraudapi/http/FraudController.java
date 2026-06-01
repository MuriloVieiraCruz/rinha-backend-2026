package br.com.murilovieira.fraudapi.http;

import br.com.murilovieira.fraudapi.domain.FraudResponse;
import br.com.murilovieira.fraudapi.domain.TransactionRequest;
import br.com.murilovieira.fraudapi.service.FraudService;
import br.com.murilovieira.fraudapi.vector.VectorStore;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class FraudController {

    private final FraudService fraudService;
    private final VectorStore  vectorStore;

    public FraudController(FraudService fraudService, VectorStore vectorStore) {
        this.fraudService = fraudService;
        this.vectorStore  = vectorStore;
    }

    @PostMapping("/fraud-score")
    public FraudResponse score(@RequestBody TransactionRequest request) {
        return fraudService.evaluate(request);
    }

    @GetMapping("/ready")
    public ResponseEntity<Void> ready() {
        return vectorStore.isReady()
                ? ResponseEntity.ok().build()
                : ResponseEntity.status(503).build();
    }
}
