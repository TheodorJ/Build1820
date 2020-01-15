#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL343.h>

enum STATE {GAME_END=0, GAME_START=1, BTN_UP=2, BTN_DOWN=3};

STATE state = GAME_END;

float dot(float[3] x, float[3] y) {
  return (x[0] * y[0]) + (x[1] * y[1]) + (x[2] * x[2]);
}

float cosine(float[3] x, float[3] y) {
  float x_dot_y = dot(x, y);
  float x_mag = dot(x, x);
  float y_mag = dot(y, y);
  return sqrt((x_dot_y * x_dot_y) / (x_mag * y_mag))
}

/* Assign a unique ID to this sensor at the same time */
/* Uncomment following line for default Wire bus      */
Adafruit_ADXL343 accel = Adafruit_ADXL343(12345);

/* NeoTrellis M4, etc.                    */
/* Uncomment following line for Wire1 bus */
//Adafruit_ADXL343 accel = Adafruit_ADXL343(12345, &Wire1);

void displayDataRate(void)
{
  Serial.print  ("Data Rate:    ");

  switch(accel.getDataRate())
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

  switch(accel.getRange())
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

void setup(void)
{
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Accelerometer Test"); Serial.println("");

  /* Initialise the sensor */
  if(!accel.begin())
  {
    /* There was a problem detecting the ADXL343 ... check your connections */
    Serial.println("Ooops, no ADXL343 detected ... Check your wiring!");
    while(1);
  }

  /* Set the range to whatever is appropriate for your project */
  accel.setRange(ADXL343_RANGE_16_G);
  // accel.setRange(ADXL343_RANGE_8_G);
  // accel.setRange(ADXL343_RANGE_4_G);
  // accel.setRange(ADXL343_RANGE_2_G);

  /* Display some basic information on this sensor */
  accel.printSensorDetails();
  displayDataRate();
  displayRange();
  Serial.println("");
}

int num_IMU_points = 0;
float[] IMU_x = float[500]; // 5 seconds of data
float[] IMU_y = float[500];
float[] IMU_z = float[500];

void loop(void)
{
  /* Get a new sensor event */
  sensors_event_t event;
  accel.getEvent(&event);
  
  switch(state){
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
      IMU_x[num_IMU_points] = event.acceleration.x;
      IMU_y[num_IMU_points] = event.acceleration.y;
      IMU_z[num_IMU_points] = event.acceleration.z;
      num_IMU_points += 1;
      // if(local_BTN_UP):
      //   spell = classify_spell(IMU_x, IMU_y, IMU_z, num_IMU_points);
      //   send(spell);  // Send the spell to the hat as a CAST <spell> message
      //   state = BTN_UP;
      break;
  }

  // if(receive_GAME_END):
  //   state = GAME_END
  
  delay(500);
}
