#include "WiFi.h"
#include "HTTPClient.h"
#include "esp_camera.h"

const char* ssid = "SEU_WIFI";
const char* password = "SENHA_WIFI";
const char* serverUrl = "http://SEU_NGROK_URL/detect/";  // exemplo: http://xxxxx.ngrok.io/detect/

// Configuração da câmera — modelo padrão do ESP32-CAM (AI Thinker)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

void setup() {
  Serial.begin(115200);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;  // 640x480
  config.jpeg_quality = 10;           // qualidade boa sem ser pesada
  config.fb_count = 1;

  // Inicializa câmera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Erro ao inicializar câmera: 0x%x", err);
    return;
  }

  // Conecta ao Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi conectado!");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Falha ao capturar imagem!");
      return;
    }

    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "multipart/form-data; boundary=ESP32CAM");

    String body = "--ESP32CAM\r\n";
    body += "Content-Disposition: form-data; name=\"file\"; filename=\"frame.jpg\"\r\n";
    body += "Content-Type: image/jpeg\r\n\r\n";

    int totalLen = body.length() + fb->len + strlen("\r\n--ESP32CAM--\r\n");
    http.collectHeaders("Content-Length", 1);
    WiFiClient *stream = http.getStreamPtr();

    http.addHeader("Content-Length", String(totalLen));
    http.POST("");

    stream->print(body);
    stream->write(fb->buf, fb->len);
    stream->print("\r\n--ESP32CAM--\r\n");

    Serial.println("📸 Frame enviado!");

    http.end();
    esp_camera_fb_return(fb);
  }
  delay(2000);  // Envia a cada 2 segundos
}
