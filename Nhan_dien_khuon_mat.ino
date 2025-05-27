#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// Thay bằng thông tin Wi-Fi của bạn
const char* ssid = "Ha Tran";
const char* password = "0961754936";
const char* API_TOKEN = "123456"; // Phải khớp với API_TOKEN trong Python

WebServer server(80);

// Biến lưu thông tin người được nhận diện
String personName = "Unknown";
String personInfo = "No info available";

// Cấu hình camera (AI-Thinker model)
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

void handleRoot() {
  String html = "<!DOCTYPE html><html>";
  html += "<head><title>ESP32-CAM Face Recognition</title></head>";
  html += "<body><h1>ESP32-CAM Face Recognition</h1>";
  html += "<img src=\"/stream\" style=\"width:640px;height:480px;\">";
  html += "<h2>Recognized Person: " + personName + "</h2>";
  html += "<p>Info: " + personInfo + "</p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleStream() {
  WiFiClient client = server.client();
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  server.sendContent(response);

  while (client.connected()) {
    camera_fb_t * fb = esp_camera_fb_get();  // ✅ Đã sửa ở đây
    if (!fb) {
      Serial.println("Camera capture failed");
      return;
    }
    response = "--frame\r\n";
    response += "Content-Type: image/jpeg\r\n\r\n";
    server.sendContent(response);
    client.write(fb->buf, fb->len);
    server.sendContent("\r\n");
    esp_camera_fb_return(fb);  // Trả lại frame sau khi dùng
    delay(10);
  }
}

void handleInfo() {
  // Kiểm tra token xác thực
  String authHeader = server.header("Authorization");
  if (authHeader != String("Bearer ") + API_TOKEN) {
    server.send(401, "text/plain", "Unauthorized");
    return;
  }

  if (server.hasArg("name") && server.hasArg("info")) {
    personName = server.arg("name");
    personInfo = server.arg("info");
    server.send(200, "text/plain", "Info updated");
  } else {
    server.send(400, "text/plain", "Missing name or info");
  }
}

void setup() {
  Serial.begin(115200);

  // Cấu hình camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
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
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 10;
  config.fb_count = 2;

  // Khởi động camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  // Kết nối Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.println(WiFi.localIP());

  // Cấu hình web server
  server.on("/", handleRoot);
  server.on("/stream", handleStream);
  server.on("/info", HTTP_POST, handleInfo);
  server.begin();
}

void loop() {
  server.handleClient();
}
