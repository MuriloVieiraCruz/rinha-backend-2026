FROM ghcr.io/graalvm/native-image-community:25 AS build
RUN microdnf install -y maven findutils

WORKDIR /app
COPY pom.xml ./pom.xml
COPY fraud-api/pom.xml ./fraud-api/pom.xml
COPY preprocess/pom.xml ./preprocess/pom.xml
RUN mvn dependency:go-offline -q -pl fraud-api -am

COPY fraud-api/src ./fraud-api/src
RUN NATIVE_IMAGE_OPTIONS="-O3 -march=compatibility" \
    mvn -pl fraud-api -am -Pnative -DskipTests package -q

FROM debian:12-slim
WORKDIR /app
COPY --from=build /app/fraud-api/target/fraud-api ./server
EXPOSE 8080
ENTRYPOINT ["./server", "-Xmx155m", "-Xms148m"]
