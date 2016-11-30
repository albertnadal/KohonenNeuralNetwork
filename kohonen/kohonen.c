#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#define MAP_WIDTH 75
#define MAP_HEIGHT 43
#define TOTAL_SAMPLES 200

typedef struct Neuron {
   unsigned int x;
   unsigned int y;
   unsigned int z;
   float weight;
} Neuron;

typedef struct BMU {
   unsigned int x_coord;
   unsigned int y_coord;
   Neuron* neuron;
   float distance;
} BMU;

typedef struct Coordinate {
   float x;
   float y;
} Coordinate;

typedef struct Sample {
   unsigned int x;
   unsigned int y;
   unsigned int z;
} Sample;

Neuron** map;
Sample* samples;
int initial_radius = 70;

unsigned int randr(unsigned int min, unsigned int max)
{
  double scaled = (double)rand()/RAND_MAX;
  return (max - min +1)*scaled + min;
}

void initialize_samples()
{
  samples = (Sample *) malloc(sizeof(Sample) * TOTAL_SAMPLES);

  for(int i=0; i<TOTAL_SAMPLES; i++) {
    samples[i].x = randr(0,255);
    samples[i].y = randr(0,255);
    samples[i].z = randr(0,255);
  }
}

void initialize_map()
{
  map = (Neuron **) malloc(sizeof(Neuron *) * MAP_WIDTH);

  int x, y;
  for(x = 0; x < MAP_WIDTH; x++) {
      map[x] = (Neuron *) malloc(sizeof(Neuron) * MAP_HEIGHT);
  }

  for(x = 0; x < MAP_WIDTH; x++) {
    for(y = 0; y < MAP_HEIGHT; y++) {
      map[x][y].x = randr(0,255);
      map[x][y].y = randr(0,255);
      map[x][y].z = randr(0,255);
      map[x][y].weight = sqrt((map[x][y].x * map[x][y].x) + (map[x][y].y * map[x][y].y) + (map[x][y].z * map[x][y].z));
    }
  }

}

Sample* pick_random_sample() {
  int i = randr(0,TOTAL_SAMPLES - 1);
  return &samples[i];
}

float distance_between_sample_and_neuron(Sample *sample, Neuron *neuron) {
  int x = sample->x - neuron->x;
  int y = sample->y - neuron->y;
  int z = sample->z - neuron->z;
  return sqrt(x*x + y*y + z*z);
}

BMU* search_bmu(Sample *sample) {
  float max_dist=999999999.9f;
  float dist = 0.0f;
  BMU *bmu = (BMU *) malloc(sizeof(BMU));

  for(int x = 0; x < MAP_WIDTH; x++) {
    for(int y = 0; y < MAP_HEIGHT; y++) {
      dist = distance_between_sample_and_neuron(sample, &map[x][y]);
      if(dist < max_dist) {
        bmu->x_coord = x;
        bmu->y_coord = y;
        bmu->neuron = &map[x][y];
        bmu->distance = dist;
        max_dist = dist;
      }
    }
  }

  return bmu;
}

float get_coordinate_distance(Coordinate *p1, Coordinate *p2) {
  float x_sub = (p1->x) - (p2->x);
  float y_sub = (p1->y) - (p2->y);
  return sqrt(x_sub*x_sub + y_sub*y_sub);
}

Coordinate* new_coordinate(float x, float y) {
  Coordinate *coordinate = malloc(sizeof(Coordinate));
  coordinate->x = x;
  coordinate->y = y;
  return coordinate;
}

void scale_neuron_at_position(int x, int y, Sample *sample, double scale) {

  Neuron *neuron = &map[x][y];
  //printf("ORIGINAL NEURON: (%d,%d,%d) | ", neuron->x, neuron->y, neuron->z);

  float neuron_prescaled_x = neuron->x * (1.0f-scale);
  float neuron_prescaled_y = neuron->y * (1.0f-scale);
  float neuron_prescaled_z = neuron->z * (1.0f-scale);

  //printf("PRESCALED NEURON: (%f,%f,%f) | ", neuron_prescaled_x, neuron_prescaled_y, neuron_prescaled_z);

  float neuron_scaled_x = (sample->x * scale) + neuron_prescaled_x;
  float neuron_scaled_y = (sample->y * scale) + neuron_prescaled_y;
  float neuron_scaled_z = (sample->z * scale) + neuron_prescaled_z;

  neuron->x = (int)neuron_scaled_x;
  neuron->y = (int)neuron_scaled_y;
  neuron->z = (int)neuron_scaled_z;

  //printf("SCALED NEURON: (%d,%d,%d)\n", neuron->x, neuron->y, neuron->z);
}

void scale_neighbors(BMU *bmu, Sample *sample, float t) {
  float iteration_radius = roundf((float)(initial_radius)*(1.0f-t));
  Coordinate *outer = new_coordinate(iteration_radius,iteration_radius);
  Coordinate *center = new_coordinate(0.0f,0.0f);
  float distance_normalized = get_coordinate_distance(center,outer);
  float distance;
  double scale;
  Neuron *neuron;
  int x_coord;
  int y_coord;

  for(float y = -iteration_radius; y<iteration_radius; y++) {
    for(float x = -iteration_radius; x<iteration_radius; x++) {
      if((y + bmu->y_coord) >= 0 && (y + bmu->y_coord) < MAP_HEIGHT && (x + bmu->x_coord)>=0 && (x + bmu->x_coord) < MAP_WIDTH) {

        outer->x = x;
        outer->y = y;
        distance = get_coordinate_distance(outer,center);
        distance /= distance_normalized;

        scale = exp(-1.0f * (pow(distance, 2.0f)) / 0.15f);
        scale /= (t*4.0f + 1.0f); // +1 is to avoid divide by 0's

        x_coord = bmu->x_coord + x;
        y_coord = bmu->y_coord + y;
        scale_neuron_at_position(x_coord, y_coord, sample, scale);

        //printf("X: %f | Y: %f | SCALE: %f | DISTANCE: %f | NORMALIZED DISTANCE: %f\n", x, y, scale, distance, distance_normalized);

      }
    }
  }

  //printf("ITERATION RADIUS: %f | NORMALIZED RADIUS: %f\n", iteration_radius, distance_normalized);
  free(outer);
  free(center);
}

void free_allocated_memory() {
  free(samples);
  for(int x = 0; x < MAP_WIDTH; x++) {
    free(map[x]);
  }
  free(map);
}

char* concat(const char *s1, const char *s2)
{
    char *result = malloc(strlen(s1)+strlen(s2)+1);//+1 for the zero-terminator
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void output_html() {

  FILE *f = fopen("test.html", "w");
  if (f == NULL)
  {
      printf("Error opening file!\n");
      exit(1);
  }

  fprintf(f, "<html><head><script>setTimeout(function(){ window.location.reload(1); }, 1000);</script></head><body>");
  fprintf(f, "<table style='border-collapse: collapse;'>");

  for(int y = 0; y < MAP_HEIGHT; y++) {
    fprintf(f, "<tr>");
    for(int x = 0; x < MAP_WIDTH; x++) {
      fprintf(f, "<td style='width:15px;height:15px;background-color:rgb(%d,%d,%d);'></td>", map[x][y].x, map[x][y].y, map[x][y].z);
    }
    fprintf(f, "</tr>");
  }

  fprintf(f, "</table>");
  fprintf(f, "</body></html>");

  fclose(f);
}

int main()
{
    int MAX_ITER = 1000;
    float T_INC = 1.0f/(float)(MAX_ITER);
    float t = 0.0f;
    BMU *bmu;

    initialize_samples();
    initialize_map();
    output_html();
    sleep(5);

    int iteration_num = 0;

    while(t < 1.0f)
    {
      Sample *sample = pick_random_sample();
//      printf("RANDOM SAMPLE X: %d | Y: %d | Z: %d\n", sample->x , sample->y, sample->z);

      bmu = search_bmu(sample);
//      printf("BMU X: %d | Y: %d | Z: %d | DIST: %f | x: %d | y: %d\n", bmu->neuron->x , bmu->neuron->y, bmu->neuron->z, bmu->distance, bmu->x_coord, bmu->y_coord);

      scale_neighbors(bmu, sample, t);

      free(bmu);
      t += T_INC;

      output_html();
      usleep(20000);
    }

    output_html();
    free_allocated_memory();

    return 0;
}
