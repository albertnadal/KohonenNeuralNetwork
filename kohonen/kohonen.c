#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>

// START Kohonen Algorithm defines and global variables
#define MAP_WIDTH 75
#define MAP_HEIGHT 43

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

// END Kohonen definitions and global variables

// START k-Means Algorithm defines and global variables

#define BIG_NUM          999999999999999999
#define LINE_SIZE        80
#define FALSE            0
#define TRUE             1
#define uint             unsigned int
#define uint64           unsigned long
#define MAX_KMEANS_ITERATIONS 100
#define datax(i)         kmeans_data_points[i * 2]
#define datay(i)         kmeans_data_points[i * 2 + 1]
#define distance(i, j)   (datax(j) - datax(i)) * (datax(j) - datax(i)) + (datay(j) - datay(i)) * (datay(j) - datay(i))
#define is_assigned(i)   point_assignments[i].assigned
#define set_assigned(i)  point_assignments[i].assigned = TRUE
#define pow2(x)          ((x) * (x))

typedef int bool;

typedef struct {
  uint    point_id;
  uint    num_points;
  uint64  sum_x;
  uint64  sum_y;

  // for calculating new centroid
  double  target_x;
  double  target_y;
  uint64  prev_distance;
  uint64  best_distance;
} centroid;

typedef struct {
  uint    centroid_id;
  uint64  distance;
  char    assigned;
} assignment;

char        *input_file_name;
uint        total_samples;
uint        num_clusters;
centroid    *centroids;
uint        *kmeans_data_points;
assignment  *point_assignments;
uint        improved = 0;
float       improvement = 0.0;

void setup_centroids();
void setup_assignments();
void update_assignments();
void update_centroids();
int gettimeofday(struct timeval *restrict tp, void *restrict tzp);

// END k-Means definitions and global variables

// START Kohonen algorithm methods

unsigned int randr(unsigned int min, unsigned int max)
{
  double scaled = (double)rand()/RAND_MAX;
  return (max - min +1)*scaled + min;
}

void load_and_initialize_samples()
{
  FILE *file;
  char line[LINE_SIZE];

  // 'kmeans_data_points' is the data structure used to store the sample points for k-means algorithm process
  // kmeans_data_points = (uint *)calloc(total_samples * 2, sizeof(uint));

  // 'samples' is the data structure used to store the sample points for Kohonen algorithm process
  samples = (Sample *) malloc(sizeof(Sample) * total_samples);

  // read data from file into array
  file = fopen(input_file_name, "rt");
  for (uint i = 0; i < total_samples; i++) {
    fgets(line, LINE_SIZE, file);
//    sscanf(line, "%d,%d", &datax(i), &datay(i));
    sscanf(line, "%d,%d,%d", &samples[i].x, &samples[i].y, &samples[i].z);
  }
  fclose(file);
}

void initialize_som_map()
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
  int i = randr(0,total_samples - 1);
  return &samples[i];
}

Sample* pick_sample(int i) {
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

void output_html(BMU *centroids) {

  bool found = FALSE;
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

      found = FALSE;
      for (int i = 0; ((i < num_clusters) && (centroids)); i++) {
        if((centroids[i].x_coord == x) && (centroids[i].y_coord == y)) {
          found = TRUE;
        }
      }

      if(found) {
        fprintf(f, "<td style='width:15px;height:15px;background-color:rgb(0,0,0);'></td>");
      } else {
        fprintf(f, "<td style='width:15px;height:15px;background-color:rgb(%d,%d,%d);'></td>", map[x][y].x, map[x][y].y, map[x][y].z);
      }

    }
    fprintf(f, "</tr>");
  }

  fprintf(f, "</table>");
  fprintf(f, "</body></html>");

  fclose(f);
}

// END Kohonen algorithm methods

// START K-Means algorithm methods

/**
 * assign data points to current centroids
 */
void update_assignments()
{
  uint      i, j, d;
  uint      nearest_centroid;
  uint64    min_dist;
  centroid  *cp, *cp_last;

  // clear out centroids
  for (cp = centroids, cp_last = centroids + num_clusters; cp < cp_last; cp++) {
    cp->num_points = 0;
    cp->sum_x = 0;
    cp->sum_y = 0;
  }

  // for each data point...
  for (i = 0; i < total_samples; i++) {
    // find nearest centroid
    for (j = 0; j < num_clusters; j++) {
      d = distance(i, centroids[j].point_id);
      if (j == 0 || min_dist > d) {
        min_dist = d;
        nearest_centroid = j;
      }
    }

    // assign point to centroid
    point_assignments[i].centroid_id = nearest_centroid;
    point_assignments[i].distance = min_dist;

    // update centroid stats
    centroids[nearest_centroid].sum_x += datax(i);
    centroids[nearest_centroid].sum_y += datay(i);
    centroids[nearest_centroid].num_points++;
  }

}

/**
 * pick new centroids based on mean coordinates in the cluster
 */
void update_centroids()
{
  uint        i, a, b;
  uint        *dp;
  uint64      d;
  float       k;
  assignment  *ap;
  centroid    *cp, *cp_last;

  // initialize
  for (cp = centroids, cp_last = centroids + num_clusters; cp < cp_last; cp++) {
    cp->target_x = cp->sum_x / cp->num_points;
    cp->target_y = cp->sum_y / cp->num_points;
    cp->prev_distance = cp->best_distance;
    cp->best_distance = BIG_NUM;
  }

  // go through points; for each respective centroid, find the
  // point closest to the target new centroid
  for (i = 0, ap = point_assignments, dp = kmeans_data_points; i < total_samples; i++, ap++, dp += 2) {
    // get my centroid
    cp = centroids + ap->centroid_id;

    // calculate distance to target
    a = *dp - cp->target_x;
    b = *(dp + 1) - cp->target_y;
    d = a * a + b * b;

    // new closest point?
    if (cp->best_distance > d) {
      cp->best_distance = d;
      cp->point_id = i;
    }
  }

  // check improvement
  improved = 0;
  improvement = 0.0;
  for (i = 0, cp = centroids; i < num_clusters; i++, cp++) {
    if (cp->prev_distance) {
      k = ((float)cp->best_distance - (float)cp->prev_distance) / (float)cp->prev_distance;
      improvement += fabs(k);
      improved++;
    }
  }

}

/**
 * allocate the array for data point to centroid assignments
 */
void setup_assignments()
{
  point_assignments = (assignment *)calloc(total_samples, sizeof(assignment));
}

/**
 * allocate the array for centroids, then pick the initial centroids
 */
void setup_centroids()
{
  uint i, j, k, x, y;
  struct timeval tv;

  // seed rand with microseconds to increase variance if called repeatedly quickly
  gettimeofday(&tv, NULL);
  srand(tv.tv_usec);

  // allocate centroids array
  centroids = (centroid *)calloc(num_clusters, sizeof(centroid));

  // pick starting centroids, ensuring uniqueness
  for (i = 0; i < num_clusters; ) {
    // grab a random data point
    j = random() % total_samples;

    // take if first one or unassigned
    if (i == 0 || !is_assigned(j)) {
      centroids[i].point_id = j;

      // mark this point (and all points like it) as assigned;
      // this traversal works because we know the data set is sorted
      x = datax(j);
      y = datay(j);

      for (k = j; k < total_samples && x == datax(k) && y == datay(k); k++)
        set_assigned(k);
      for (k = j; k-- > 0 && x == datax(k) && y == datay(k); )
        set_assigned(k);

      // on to next centroid
      i++;
    }

    // done if all assigned
    for (j = 0; j < total_samples; j++)
      if (!is_assigned(j)) break;
    if (j >= total_samples) break;
  }

  // it's possible that the actual number of clusters is less than asked for
  if (i < num_clusters)
    num_clusters = i;
}

void search_clusters_in_bmu()
{
  uint        i, j, x, y;
  double      distance_mean, distance_sum;
  float       average_improvement;
  assignment  *ap, *ap_last;
  centroid    *cp;

  setup_assignments();
  setup_centroids();

  for (uint i = 0; i < MAX_KMEANS_ITERATIONS; i++) {
    update_assignments();
    update_centroids();
    if (improved) {
      average_improvement = improvement / improved;

      // done if average improvement is less than 0.1%
      if (average_improvement < 0.001)
        break;
    }
  }

  // calculate standard deviation of point-to-centroid distances
  for (i = 0, cp = centroids; i < num_clusters; i++, cp++) {
    // centroid coords
    x = datax(cp->point_id);
    y = datay(cp->point_id);

    if (cp->num_points <= 1) {
      printf("%d\t%d\t%d\t0.0\n", x, y, cp->num_points);
    } else {
      // calculate mean distance
      distance_sum = 0.0;
      for (j = 0, ap = point_assignments, ap_last = point_assignments + total_samples; ap < ap_last; j++, ap++) {
        if (ap->centroid_id != i) continue;
        distance_sum += sqrt(ap->distance);
      }
      distance_mean = distance_sum / cp->num_points;

      // calculate summation for sample variance
      distance_sum = 0.0;
      for (ap = point_assignments, ap_last = point_assignments + total_samples; ap < ap_last; ap++) {
        if (ap->centroid_id != i) continue;
        distance_sum += pow2(sqrt(ap->distance) - distance_mean);
      }

      printf("Centroid: (%d,%d)\tCluster size(bmu's): %d\tAvg. distance: %.1lf\n", x, y, cp->num_points, sqrt(distance_sum / (cp->num_points - 1)));
    }
  }

}

// END K-Means algorithm methods

int main(int argc, char **argv)
{
    int MAX_ITER = 1000;
    float T_INC = 1.0f/(float)(MAX_ITER);
    float t = 0.0f;
    BMU *bmu;
    BMU *cluster_centroid_bmus;
    BMU *centroid_bmu;
    centroid  *cp;
    uint i;

    // usage: ./kohonen file_with_samples total_samples num_clusters
    input_file_name = argv[1];
    total_samples = atoi(argv[2]);
    num_clusters = atoi(argv[3]);

    // seed random
    srand(time(NULL));

    load_and_initialize_samples();

    initialize_som_map();
    output_html(0);

    int iteration_num = 0;

    while(t < 1.0f)
    {
      Sample *sample = pick_random_sample();
      bmu = search_bmu(sample); // Best Match Unit
      scale_neighbors(bmu, sample, t);
      free(bmu);

      t += T_INC;

      output_html(0);

      //usleep(100000);
    }


    // Collect the final BMU coordinates in the SOM map
    kmeans_data_points = (uint *)calloc(total_samples * 2, sizeof(uint));
    for (uint i = 0; i < total_samples; i++) {
	Sample *sample = pick_sample(i);	
	bmu = search_bmu(sample); // Best Match Unit
        datax(i) = bmu->x_coord;
	datay(i) = bmu->y_coord;
	free(bmu);
    }

    // Search clusters from BMU coordinates
    search_clusters_in_bmu();

    // Prepare structure for painting clusters centroids
    cluster_centroid_bmus = (BMU *)malloc(sizeof(BMU) * num_clusters);
    for (i = 0, cp = centroids; i < num_clusters; i++, cp++) {
        cluster_centroid_bmus[i].x_coord = datax(cp->point_id);
        cluster_centroid_bmus[i].y_coord = datay(cp->point_id);
    	free(centroid_bmu);
    }

    output_html(cluster_centroid_bmus);

    free(cluster_centroid_bmus);
    free(kmeans_data_points);
    free_allocated_memory();

    return 0;
}
