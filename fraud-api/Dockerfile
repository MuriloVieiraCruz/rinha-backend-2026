# Stage 1: Build the IVF index (refs.bin) from references.json.gz
FROM ghcr.io/graalvm/native-image-community:25 AS indexer
RUN microdnf install -y maven findutils
WORKDIR /build
COPY pom.xml .
COPY fraud-api/pom.xml fraud-api/pom.xml
COPY preprocess/pom.xml preprocess/pom.xml
RUN mvn dependency:go-offline -q -pl preprocess -am
COPY preprocess/src preprocess/src
COPY resources/references.json.gz /data/references.json.gz
RUN mvn -pl preprocess -am -DskipTests package -q && \
    $JAVA_HOME/bin/java -Xmx2g -jar preprocess/target/preprocess.jar \
        /data/references.json.gz /tmp/refs.bin

# Stage 2: Build the native image
FROM ghcr.io/graalvm/native-image-community:25 AS build
RUN microdnf install -y maven findutils
WORKDIR /build
COPY pom.xml .
COPY fraud-api/pom.xml fraud-api/pom.xml
COPY preprocess/pom.xml preprocess/pom.xml
RUN mvn dependency:go-offline -q -pl fraud-api -am
COPY fraud-api/src fraud-api/src
RUN NATIVE_IMAGE_OPTIONS="-O3 -march=compatibility" \
    mvn -pl fraud-api -am -Pnative -DskipTests package -q

# Stage 3: Minimal runtime
FROM debian:12-slim
WORKDIR /app
COPY --from=build  /build/fraud-api/target/fraud-api ./server
COPY --from=indexer /tmp/refs.bin ./refs.bin
ENTRYPOINT ["./server", "-Xmx148m", "-Xms140m"]
