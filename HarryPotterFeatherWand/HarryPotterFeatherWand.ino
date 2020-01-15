#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL343.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ---------------------- //
// VARIABLES & PARAMETERS //
// ---------------------- //

// Game state
enum state_t {GAME_END = 0, GAME_START = 1, BTN_UP = 2, BTN_DOWN = 3};

state_t state = GAME_END;

// Gesture detection
enum spell_t {ERR = -1, LEFT = 0, RIGHT = 1, UP = 2, DOWN = 3};
String spell_names[4] = {"Left", "Right", "Up", "Down"};
// Minimum acceleration to trigger a spell (m/s^2)
const float min_spell_accel = 2.5;
#define COS_45 (0.7071)

// Bluetooth
const char *ble_name = "MagicWand";
BLEServer * ble_server = NULL;
BLECharacteristic* ble_spell_characteristic = NULL;
volatile bool ble_dev_connected = false;
bool ble_old_dev_connected = false;

#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"


// Accelerometer
/* Assign a unique ID to this sensor */
Adafruit_ADXL343 accel = Adafruit_ADXL343(12345);

// ---------------- //
// HELPER FUNCTIONS //
// ---------------- //

float dot3(float x[3], float y[3]) {
  return (x[0] * y[0]) + (x[1] * y[1]) + (x[2] * x[2]);
}

float dot2(float x[2], float y[2]) {
  return (x[0] * y[0]) + (x[1] * y[1]);
}

// Returns the cosine of the angle between two vectors
float vec_cos3(float x[3], float y[3]) {
  float x_dot_y = dot3(x, y);
  float x_mag_sq = dot3(x, x);
  float y_mag_sq = dot3(y, y);
  return x_dot_y / sqrt(x_mag_sq * y_mag_sq);
}
float vec_cos2(float x[2], float y[2]) {
  float x_dot_y = dot2(x, y);
  float x_mag_sq = dot2(x, x);
  float y_mag_sq = dot2(y, y);
  return x_dot_y / sqrt(x_mag_sq * y_mag_sq);
}

void displayDataRate(void)
{
  Serial.print  ("Data Rate:    ");

  switch (accel.getDataRate())
  {
    case ADXL343_DATARATE_3200_HZ:
      Serial.print  ("3200 ");
      break;
    case ADXL343_DATARATE_1600_HZ:
      Serial.print  ("1600 ");
      break;
    case ADXL343_DATARATE_800_HZ:
      Serial.print  ("800 ");
      break;
    case ADXL343_DATARATE_400_HZ:
      Serial.print  ("400 ");
      break;
    case ADXL343_DATARATE_200_HZ:
      Serial.print  ("200 ");
      break;
    case ADXL343_DATARATE_100_HZ:
      Serial.print  ("100 ");
      break;
    case ADXL343_DATARATE_50_HZ:
      Serial.print  ("50 ");
      break;
    case ADXL343_DATARATE_25_HZ:
      Serial.print  ("25 ");
      break;
    case ADXL343_DATARATE_12_5_HZ:
      Serial.print  ("12.5 ");
      break;
    case ADXL343_DATARATE_6_25HZ:
      Serial.print  ("6.25 ");
      break;
    case ADXL343_DATARATE_3_13_HZ:
      Serial.print  ("3.13 ");
      break;
    case ADXL343_DATARATE_1_56_HZ:
      Serial.print  ("1.56 ");
      break;
    case ADXL343_DATARATE_0_78_HZ:
      Serial.print  ("0.78 ");
      break;
    case ADXL343_DATARATE_0_39_HZ:
      Serial.print  ("0.39 ");
      break;
    case ADXL343_DATARATE_0_20_HZ:
      Serial.print  ("0.20 ");
      break;
    case ADXL343_DATARATE_0_10_HZ:
      Serial.print  ("0.10 ");
      break;
    default:
      Serial.print  ("???? ");
      break;
  }
  Serial.println(" Hz");
}

void displayRange(void)
{
  Serial.print  ("Range:         +/- ");

  switch (accel.getRange())
  {
    case ADXL343_RANGE_16_G:
      Serial.print  ("16 ");
      break;
    case ADXL343_RANGE_8_G:
      Serial.print  ("8 ");
      break;
    case ADXL343_RANGE_4_G:
      Serial.print  ("4 ");
      break;
    case ADXL343_RANGE_2_G:
      Serial.print  ("2 ");
      break;
    default:
      Serial.print  ("?? ");
      break;
  }
  Serial.println(" g");
}

// BLE connection/disconnection callbacks
class ble_server_callbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      ble_dev_connected = true;
    };

    void onDisconnect(BLEServer* pServer) {
      ble_dev_connected = false;
    }
};

void display_ble_status(void)
{
  Serial.println(ble_dev_connected ? "[BLE] Connected!" : "[BLE] Disconnected");
}

// Check for BLE disconnection/connection
bool check_ble_status(void)
{
  // Check if BLE has disconnected
  if (!ble_dev_connected && ble_old_dev_connected) {
    display_ble_status();
    delay(500); // give the bluetooth stack the chance to get things ready
    ble_server->startAdvertising(); // restart advertising
    Serial.println("[BLE] restarting advertising");
    ble_old_dev_connected = ble_dev_connected;
  }

  // Check if BLE has reconnected
  if (ble_dev_connected && !ble_old_dev_connected) {
    display_ble_status();
    ble_old_dev_connected = ble_dev_connected;
  }

  return ble_dev_connected;
}

// Send a newly detected spell to connected device (raspberry Pi) via BLE notification
void ble_send_value(const char *spell)
{
  if (check_ble_status()) {
    ble_spell_characteristic->setValue((uint8_t*)spell, strlen(spell));
    ble_spell_characteristic->notify();
    Serial.print("[BLE] Sent message: ");
    Serial.println(spell);
    delay(3); // To prevent congestion. TODO can remove if delay is guaranteed elsewise
  }
}

// Detect the direction of an acceleration vector, equivalently which spell this is.
// Gravity should already be removed from p3.
spell_t accel_dir(float gravity3[3], float p3[3])
{
  // Only use the Y and Z of both vectors
  float gravity2[2] = {gravity3[1], gravity3[2]};
  float p[2] = {p3[1], p3[2]};

  // Cosine of angle between gravity and p
  float cos_vec = vec_cos2(gravity2, p);

  // Prediction of what spell this is
  spell_t pred;

  // (x component of) cross product between gravity and p3
  if (cos_vec > COS_45) {
    // Large cos means along direction of gravity, and measured gravity actually points against gravity
    pred = UP;
  } else if (cos_vec < -1 * COS_45) {
    // Large negative cos means against dir of gravity, flipped as before
    pred = DOWN;
  } else {
    // Middling cos means angle is left or right, use cross product to determine
    float cross = gravity2[0] * p[1] - gravity2[1] * p[0];

    if (cross > 0) {
      pred = RIGHT;
    } else {
      pred = LEFT;
    }
  }

  return pred;
}

// -------------- //
// Initialization //
// -------------- //

// Initialize accelerometer. Hang until complete.
void init_accelerometer(void)
{
  /* Initialise the sensor */
  while (!accel.begin())
  {
    /* There was a problem detecting the ADXL343 ... check your connections */
    Serial.println("Ooops, no ADXL343 detected ... Check your wiring!");
    // TODO: flash error LED
    delay(500);
  }

  /* Set the range to whatever is appropriate for your project */
  accel.setRange(ADXL343_RANGE_16_G);

  /* Display some basic information on this sensor */
  accel.printSensorDetails();
  displayDataRate();
  displayRange();
  Serial.println("");
}

// Initialize BLE connection, do not wait for connections
void init_ble(void)
{
  BLEDevice::init(ble_name);

  Serial.print("[BLE] initialized with MAC ");
  Serial.println(BLEDevice::getAddress().toString().c_str());

  // Create the BLE Server
  ble_server = BLEDevice::createServer();
  ble_server->setCallbacks(new ble_server_callbacks());

  // Create the BLE Service
  BLEService *ble_service = ble_server->createService(SERVICE_UUID);

  // Create a BLE Characteristic
  ble_spell_characteristic = ble_service->createCharacteristic(
                               CHARACTERISTIC_UUID,
                               BLECharacteristic::PROPERTY_READ   |
                               BLECharacteristic::PROPERTY_WRITE  |
                               BLECharacteristic::PROPERTY_NOTIFY |
                               BLECharacteristic::PROPERTY_INDICATE
                             );

  // https://www.bluetooth.com/specifications/gatt/viewer?attributeXmlFile=org.bluetooth.descriptor.gatt.client_characteristic_configuration.xml
  // Create a BLE Descriptor
  ble_spell_characteristic->addDescriptor(new BLE2902());

  // Start the service
  ble_service->start();

  // Start advertising
  BLEAdvertising *ble_advertising = BLEDevice::getAdvertising();
  ble_advertising->addServiceUUID(SERVICE_UUID);
  ble_advertising->setScanResponse(false);
  ble_advertising->setMinPreferred(0x0);  // set value to 0x00 to not advertise this parameter
  BLEDevice::startAdvertising();
  Serial.println("[BLE] Waiting for a client connection...");
}

// Hang until BLE is connected
void connect_ble(void)
{
  // Hang until connection callback changes status
  while (!ble_dev_connected);

  display_ble_status();
}

void setup(void)
{
  Serial.begin(115200);
  Serial.println("Welcome to the MagikWand 7000(TM)(C)(R)!"); Serial.println("");

  init_accelerometer();
  init_ble();
}

// --------- //
// MAIN GAME //
// --------- //


void loop(void)
{
  // Read accelerometer
  sensors_event_t event;
  accel.getEvent(&event);

  switch (state) {
    case GAME_END:
      // if(local_BTN_DOWN):
      //   send("BTN_DOWN")
      // if(local_BTN_UP):
      //   send("BTN_UP")
      // if(receive_GAME_START):
      //   state = GAME_START
      break;
    case GAME_START:
      // if(local_BTN_UP):
      //   state = BTN_UP
      break;
    case BTN_UP:
      // if(local_BTN_DOWN):
      //   state = BTN_DOWN
      break;
    case BTN_DOWN:
      //IMU_x[num_IMU_points] = event.acceleration.x;
      //IMU_y[num_IMU_points] = event.acceleration.y;
      //IMU_z[num_IMU_points] = event.acceleration.z;
      //num_IMU_points += 1;
      // if(local_BTN_UP):
      //   spell = classify_spell(IMU_x, IMU_y, IMU_z, num_IMU_points);
      //   send(spell);  // Send the spell to the hat as a CAST <spell> message
      //   state = BTN_UP;
      break;
  }

  // TODO use a real measurement of gravity, this is temporary
  float gravity[3] = {0.0, 0.0, 10.0};
  
  float accel[3] = {event.acceleration.x - gravity[0],
                    event.acceleration.y - gravity[1],
                    event.acceleration.z - gravity[2]
                   };

  // Get magnitude of this acceleration in YZ plane
  float mag_p = sqrt(sq(accel[1]) + sq(accel[2]));
  if (mag_p > min_spell_accel)
  {
    spell_t spell = accel_dir(gravity, accel);
    String spell_name = spell_names[spell];
    ble_send_value(spell_name.c_str());
    Serial.print("Spell cast: ");
    Serial.println(spell_name);
    delay(1000);
  }

  // if(receive_GAME_END):
  //   state = GAME_END

  // Check for BLE disconnection
  // TODO: handle disconnection event
  check_ble_status();
  delay(500);
}
