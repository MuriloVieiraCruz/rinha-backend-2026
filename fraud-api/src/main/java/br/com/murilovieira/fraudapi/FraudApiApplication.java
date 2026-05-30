package br.com.murilovieira.fraudapi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class FraudApiApplication {

    public static void main(String[] args) {
        SpringApplication.run(FraudApiApplication.class, args);
    }
}
