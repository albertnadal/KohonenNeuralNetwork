#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <thread>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <shader_s.h>

typedef short int16;
typedef int int32;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned char uchar;

const uint32 SCR_WIDTH = 800;
const uint32 SCR_HEIGHT = 450;

pthread_t kohonenMainThreadId;

#define MAP_WIDTH 40
#define MAP_HEIGHT 30
#define TOTAL_COMPONENTS 3
#define INITIAL_TRAINING_ITERATIONS_PER_EPOCH 1500
#define TOTAL_EPOCHS 8
#define INITIAL_RADIUS 14
#define INITIAL_LEARNING_RULE 0.9f
#define pow2(x) ((x) * (x))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define uint unsigned int
#define uint64 unsigned long

typedef struct Neuron {
   unsigned int* components;
} Neuron;

typedef struct BMU {
   unsigned int x_coord;
   unsigned int y_coord;
} BMU;

typedef struct CentroidBMU {
   unsigned int x_coord;
   unsigned int y_coord;
   int count;
} CentroidBMU;

typedef struct Coordinate {
   float x;
   float y;
} Coordinate;

typedef struct IntCoordinate {
   int x;
   int y;
} IntCoordinate;

typedef struct Sample {
   unsigned int* components;
} Sample;

typedef struct Sprite {
   uint32 texture_id;
   uint32 texture_ref;
   unsigned char *texture;
   uint16 *vertices_buffer;
   float *uvs_buffer;
   uint32 VBO;
   uint32 VAO;
   uint32 UBO;
   uint32 width;
   uint32 height;
   uint32 x;
   uint32 y;
} Sprite;

Neuron** map;
Sample* samples;
float learning_rule = INITIAL_LEARNING_RULE;
float radius = INITIAL_RADIUS;
uint32 epoch = 0;
uint32 iteration = 0;
uint32 iterations_per_epoch = INITIAL_TRAINING_ITERATIONS_PER_EPOCH;
uint32 total_samples = 0;
IntCoordinate highlighted_positions[10000];
uint32 total_scaled_neighbours = 0;

GLFWwindow* window;
Shader* shader;
Sprite sprite[4];

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void process_input(GLFWwindow *window);

void load_and_initialize_samples()
{
  total_samples = 27;
  printf("\n\nTotal samples: %d\n\n", total_samples);

  // 'samples' is the data structure used to store the sample points for Kohonen algorithm process
  samples = (Sample *) malloc(sizeof(Sample) * total_samples);

  for(int i = 0; i < total_samples; i++) {
    samples[i].components = (unsigned int *) malloc(sizeof(unsigned int) * TOTAL_COMPONENTS);
  }

  samples[0].components[0] = 3;
  samples[0].components[1] = 255;
  samples[0].components[2] = 0;

  samples[1].components[0] = 0;
  samples[1].components[1] = 247;
  samples[1].components[2] = 3;

  samples[2].components[0] = 0;
  samples[2].components[1] = 252;
  samples[2].components[2] = 5;

  samples[3].components[0] = 3;
  samples[3].components[1] = 239;
  samples[3].components[2] = 8;

  samples[4].components[0] = 0;
  samples[4].components[1] = 232;
  samples[4].components[2] = 3;

  samples[5].components[0] = 3;
  samples[5].components[1] = 255;
  samples[5].components[2] = 0;

  samples[6].components[0] = 0;
  samples[6].components[1] = 250;
  samples[6].components[2] = 5;

  samples[7].components[0] = 255;
  samples[7].components[1] = 3;
  samples[7].components[2] = 0;

  samples[8].components[0] = 247;
  samples[8].components[1] = 3;
  samples[8].components[2] = 0;

  samples[9].components[0] = 252;
  samples[9].components[1] = 5;
  samples[9].components[2] = 0;

  samples[10].components[0] = 239;
  samples[10].components[1] = 8;
  samples[10].components[2] = 3;

  samples[11].components[0] = 232;
  samples[11].components[1] = 0;
  samples[11].components[2] = 3;

  samples[12].components[0] = 255;
  samples[12].components[1] = 3;
  samples[12].components[2] = 3;

  samples[13].components[0] = 249;
  samples[13].components[1] = 5;
  samples[13].components[2] = 3;

  samples[14].components[0] = 247;
  samples[14].components[1] = 3;
  samples[14].components[2] = 5;

  samples[15].components[0] = 234;
  samples[15].components[1] = 8;
  samples[15].components[2] = 3;

  samples[16].components[0] = 3;
  samples[16].components[1] = 0;
  samples[16].components[2] = 255;

  samples[17].components[0] = 3;
  samples[17].components[1] = 0;
  samples[17].components[2] = 247;

  samples[18].components[0] = 5;
  samples[18].components[1] = 0;
  samples[18].components[2] = 252;

  samples[19].components[0] = 8;
  samples[19].components[1] = 3;
  samples[19].components[2] = 239;

  samples[20].components[0] = 0;
  samples[20].components[1] = 3;
  samples[20].components[2] = 232;

  samples[21].components[0] = 3;
  samples[21].components[1] = 3;
  samples[21].components[2] = 255;

  samples[22].components[0] = 5;
  samples[22].components[1] = 3;
  samples[22].components[2] = 249;

  samples[23].components[0] = 3;
  samples[23].components[1] = 5;
  samples[23].components[2] = 247;

  samples[24].components[0] = 8;
  samples[24].components[1] = 3;
  samples[24].components[2] = 234;

  samples[25].components[0] = 3;
  samples[25].components[1] = 5;
  samples[25].components[2] = 247;

  samples[26].components[0] = 5;
  samples[26].components[1] = 3;
  samples[26].components[2] = 234;
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
      map[x][y].components = (unsigned int *) malloc(sizeof(unsigned int) * TOTAL_COMPONENTS);
      for(int i=0; i<TOTAL_COMPONENTS;i++) {
        map[x][y].components[i] = rand() % 255;
      }
    }
  }
}

Sample* pick_random_sample() {
  int i = rand() % total_samples;
  return &samples[i];
}

uint distance_between_sample_and_neuron(Sample *sample, Neuron *neuron) {
  unsigned int euclidean_distance = 0;
  unsigned int component_diff;

  for(int i = 0; i < TOTAL_COMPONENTS; i++) {
    component_diff = sample->components[i] - neuron->components[i];
    euclidean_distance += pow2(component_diff);
  }

  return euclidean_distance;
  //return sqrt(euclidean_distance);
}

BMU* search_bmu(Sample *sample) {
  uint max_dist=999999999;
  uint dist = 0;
  BMU *bmu = (BMU *) malloc(sizeof(BMU));

  for(int x = 0; x < MAP_WIDTH; x++) {
    for(int y = 0; y < MAP_HEIGHT; y++) {
      dist = distance_between_sample_and_neuron(sample, &map[x][y]);
      if(dist < max_dist) {
        bmu->x_coord = x;
        bmu->y_coord = y;
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
  Coordinate *coordinate = (Coordinate *)malloc(sizeof(Coordinate));
  coordinate->x = x;
  coordinate->y = y;
  return coordinate;
}

void scale_neuron_at_position(int x, int y, Sample *sample, double scale) {
  float neuron_prescaled, neuron_scaled;
  Neuron *neuron = &map[x][y];

  for(int i=0; i<TOTAL_COMPONENTS; i++) {
    neuron_prescaled = neuron->components[i] * (1.0f-scale);
    neuron_scaled = (sample->components[i] * scale) + neuron_prescaled;
    neuron->components[i] = (int)neuron_scaled;
  }
}

void scale_neighbors(BMU *bmu, Sample *sample, float iteration_radius, float learning_rule) {
  Coordinate *outer = new_coordinate(iteration_radius,iteration_radius);
  Coordinate *center = new_coordinate(0.0f,0.0f);
  float distance;
  double scale;
  int x_coord;
  int y_coord;
  total_scaled_neighbours = 0;

  for(float y = -iteration_radius; y<iteration_radius; y++) {
    for(float x = -iteration_radius; x<iteration_radius; x++) {
      if((y + bmu->y_coord) >= 0 && (y + bmu->y_coord) < MAP_HEIGHT && (x + bmu->x_coord)>=0 && (x + bmu->x_coord) < MAP_WIDTH) {
        outer->x = x;
        outer->y = y;
        distance = get_coordinate_distance(outer,center);
        if(distance < iteration_radius) {

          scale = learning_rule * exp(-10.0f * (distance * distance) / (iteration_radius * iteration_radius));

          x_coord = bmu->x_coord + x;
          y_coord = bmu->y_coord + y;

          highlighted_positions[total_scaled_neighbours].x = x_coord;
          highlighted_positions[total_scaled_neighbours].y = y_coord;
          total_scaled_neighbours++;

          scale_neuron_at_position(x_coord, y_coord, sample, scale);
          usleep(10);
        }
      }
    }
  }

  free(outer);
  free(center);
}

void free_allocated_memory() {
  for(int i = 0; i < total_samples; i++) {
    free(samples[i].components);
  }
  free(samples);

  for(int x = 0; x < MAP_WIDTH; x++) {
    for(int y = 0; y < MAP_HEIGHT; y++) {
      free(map[x][y].components);
    }
    free(map[x]);
  }
  free(map);
}

void update_textures()
{
  for(uint s = 0; s < 4; s++) {
    for(uint y = 0; y < MAP_HEIGHT; y++) {
      for(uint x = 0; x < MAP_WIDTH; x++) {
        if((s == 0) || (s == 1)) {
          sprite[s].texture[(y * MAP_WIDTH * sizeof(GL_RGBA)) + (x * sizeof(GL_RGBA))] = (int)(map[x][y].components[0]);
        } else {
          sprite[s].texture[(y * MAP_WIDTH * sizeof(GL_RGBA)) + (x * sizeof(GL_RGBA))] = 0;
        }

        if((s == 0) || (s == 2)) {
          sprite[s].texture[(y * MAP_WIDTH * sizeof(GL_RGBA)) + (x * sizeof(GL_RGBA)) + 1] = (int)(map[x][y].components[1]);
        } else {
          sprite[s].texture[(y * MAP_WIDTH * sizeof(GL_RGBA)) + (x * sizeof(GL_RGBA)) + 1] = 0;
        }

        if((s == 0) || (s == 3)) {
          sprite[s].texture[(y * MAP_WIDTH * sizeof(GL_RGBA)) + (x * sizeof(GL_RGBA)) + 2] = (int)(map[x][y].components[2]);
        } else {
          sprite[s].texture[(y * MAP_WIDTH * sizeof(GL_RGBA)) + (x * sizeof(GL_RGBA)) + 2] = 0;
        }

        sprite[s].texture[(y * MAP_WIDTH * sizeof(GL_RGBA)) + (x * sizeof(GL_RGBA)) + 3] = 255;
      }
    }
  }

  for(uint e=0; e<total_scaled_neighbours; e++) {
    sprite[0].texture[(highlighted_positions[e].y * MAP_WIDTH * sizeof(GL_RGBA)) + (highlighted_positions[e].x * sizeof(GL_RGBA)) + 3] = 150;
  }
  usleep(1);
}

static void* kohonenMainThreadFunc(void* v)
{
  load_and_initialize_samples();
  BMU *bmu;
  Sample *sample;

  // random seed
  srand(time(NULL));

  initialize_som_map();
  update_textures();

  epoch = 0;
  while(epoch < TOTAL_EPOCHS)
  {
     radius = max(1.0f, INITIAL_RADIUS * exp(-100.0f * (epoch * epoch) / (TOTAL_EPOCHS * TOTAL_EPOCHS)));
     learning_rule = max(0.015f, INITIAL_LEARNING_RULE * exp(-10.0f * (epoch * epoch) / (TOTAL_EPOCHS * TOTAL_EPOCHS)));
     iterations_per_epoch = (epoch == 0) ? INITIAL_TRAINING_ITERATIONS_PER_EPOCH : (int)(iterations_per_epoch * 2.0f);
     epoch++;

     printf("EPOCH %d/%d | TOTAL ITERATIONS: %d | RADIUS: %.2f | LEARNING RULE: %.4f\n", epoch, TOTAL_EPOCHS, iterations_per_epoch, radius, learning_rule);

     iteration = 0;
     while(iteration < iterations_per_epoch)
     {
       sample = pick_random_sample();
       bmu = search_bmu(sample); // Best Match Unit
       scale_neighbors(bmu, sample, radius, learning_rule);
       free(bmu);

       iteration++;
       update_textures();
       usleep(10);
     }
  }

  total_scaled_neighbours = 0;
  update_textures();

  free_allocated_memory();
  return 0;
}

int main(int argc, char **argv)
{
  char window_title[256];
  window_title[255] = '\0';

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

  window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Kohonen", NULL, NULL);
  if (window == NULL)
  {
          std::cout << "Failed to create GLFW window" << std::endl;
          glfwTerminate();
          return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyboard_callback);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
          std::cout << "Failed to initialize GLAD" << std::endl;
          return -1;
  }

  shader = new Shader("shader.vs", "shader.fs");

  Sprite map_sprite = { .texture_id = 1, .texture_ref = 0, .texture = nullptr, .vertices_buffer = nullptr, .uvs_buffer = nullptr, .VBO = 0, .VAO = 0, .UBO = 0, .width = 1080, .height = 810, .x = 0, .y = 0 };
  Sprite component_a_map = { .texture_id = 1, .texture_ref = 0, .texture = nullptr, .vertices_buffer = nullptr, .uvs_buffer = nullptr, .VBO = 0, .VAO = 0, .UBO = 0, .width = 360, .height = 270, .x = 1080, .y = 540  };
  Sprite component_b_map = { .texture_id = 1, .texture_ref = 0, .texture = nullptr, .vertices_buffer = nullptr, .uvs_buffer = nullptr, .VBO = 0, .VAO = 0, .UBO = 0, .width = 360, .height = 270, .x = 1080, .y = 270  };
  Sprite component_c_map = { .texture_id = 1, .texture_ref = 0, .texture = nullptr, .vertices_buffer = nullptr, .uvs_buffer = nullptr, .VBO = 0, .VAO = 0, .UBO = 0, .width = 360, .height = 270, .x = 1080, .y = 0  };

  sprite[0] = map_sprite;
  sprite[1] = component_a_map;
  sprite[2] = component_b_map;
  sprite[3] = component_c_map;

  for(int s=0; s<4; s++) {
    glGenTextures(sprite[s].texture_id, &sprite[s].texture_ref);
    glBindTexture(GL_TEXTURE_2D, sprite[s].texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    sprite[s].texture = (unsigned char *)malloc(sizeof(GL_RGBA) * MAP_WIDTH * MAP_HEIGHT);

    for(int i=0; i < MAP_WIDTH*MAP_HEIGHT*sizeof(GL_RGBA); i+=sizeof(GL_RGBA)) {
      sprite[s].texture[i] = rand() % 255;
      sprite[s].texture[i+1] = rand() % 255;
      sprite[s].texture[i+2] = rand() % 255;
      sprite[s].texture[i+3] = 255;
    }

    sprite[s].vertices_buffer = (uint16 *)calloc(1 * 12, sizeof(uint16));
    sprite[s].uvs_buffer = (float *)calloc(1 * 12, sizeof(float));

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, MAP_WIDTH, MAP_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, sprite[s].texture);
    glGenerateMipmap(GL_TEXTURE_2D);

    // top right
    sprite[s].vertices_buffer[0 * 12] = sprite[s].x + sprite[s].width;
    sprite[s].vertices_buffer[0 * 12 + 1] = sprite[s].y + 0;

    // bottom right
    sprite[s].vertices_buffer[0 * 12 + 2] = sprite[s].x + sprite[s].width;
    sprite[s].vertices_buffer[0 * 12 + 3] = sprite[s].y + sprite[s].height;

    // top left
    sprite[s].vertices_buffer[0 * 12 + 4] = sprite[s].x + 0;
    sprite[s].vertices_buffer[0 * 12 + 5] = sprite[s].y + 0;

    // bottom right
    sprite[s].vertices_buffer[0 * 12 + 6] = sprite[s].x + sprite[s].width;
    sprite[s].vertices_buffer[0 * 12 + 7] = sprite[s].y + sprite[s].height;

    // bottom left
    sprite[s].vertices_buffer[0 * 12 + 8] = sprite[s].x + 0;
    sprite[s].vertices_buffer[0 * 12 + 9] = sprite[s].y + sprite[s].height;

    // top left
    sprite[s].vertices_buffer[0 * 12 + 10] = sprite[s].x + 0;
    sprite[s].vertices_buffer[0 * 12 + 11] = sprite[s].y + 0;

    // top right
    sprite[s].uvs_buffer[0 * 12] = 1.0f;
    sprite[s].uvs_buffer[0 * 12 + 1] = 1.0f;

    // bottom right
    sprite[s].uvs_buffer[0 * 12 + 2] = 1.0f;
    sprite[s].uvs_buffer[0 * 12 + 3] = 0.0f;

    // top left
    sprite[s].uvs_buffer[0 * 12 + 4] = 0.0f;
    sprite[s].uvs_buffer[0 * 12 + 5] = 1.0f;

    // bottom right
    sprite[s].uvs_buffer[0 * 12 + 6] = 1.0f;
    sprite[s].uvs_buffer[0 * 12 + 7] = 0.0f;

    // bottom left
    sprite[s].uvs_buffer[0 * 12 + 8] = 0.0f;
    sprite[s].uvs_buffer[0 * 12 + 9] = 0.0f;

    // top left
    sprite[s].uvs_buffer[0 * 12 + 10] = 0.0f;
    sprite[s].uvs_buffer[0 * 12 + 11] = 1.0;

    glGenVertexArrays(sprite[s].texture_id, &sprite[s].VAO);
    glGenBuffers(sprite[s].texture_id, &sprite[s].VBO);
    glGenBuffers(sprite[s].texture_id, &sprite[s].UBO);
    glBindVertexArray(sprite[s].VAO);

    glBindBuffer(GL_ARRAY_BUFFER, sprite[s].VBO);
    glBufferData(GL_ARRAY_BUFFER, 1 * 12 * sizeof(uint16), sprite[s].vertices_buffer, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_UNSIGNED_SHORT, GL_FALSE, 2 * sizeof(uint16), 0);

    glBindBuffer(GL_ARRAY_BUFFER, sprite[s].UBO);
    glBufferData(GL_ARRAY_BUFFER, 1 * 12 * sizeof(float), sprite[s].uvs_buffer, GL_DYNAMIC_DRAW /*GL_STATIC_DRAW*/);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_TRUE, 2 * sizeof(float), 0);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    shader->use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sprite[s].texture_ref);
    glBindVertexArray(sprite[s].VAO);
  }

  glClearColor(255.0f, 255.0f, 255.0f, 1.0f);

  pthread_create(&kohonenMainThreadId, NULL, kohonenMainThreadFunc, 0);

  while (!glfwWindowShouldClose(window))
  {
          process_input(window);
          glfwPollEvents();

          // Render
          glClear(GL_COLOR_BUFFER_BIT);
          glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
          //glFrontFace(GL_CCW);
          glEnable(GL_BLEND);
          glEnable(GL_CULL_FACE);
          glDisable(GL_DEPTH_TEST);
          glDisable(GL_SCISSOR_TEST);

          for(int s=0; s<4; s++) {
            glDeleteTextures(sprite[s].texture_id, &sprite[s].texture_ref);
            glGenTextures(sprite[s].texture_id, &sprite[s].texture_ref);
            glBindTexture( GL_TEXTURE_2D, sprite[s].texture_id);
            glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, MAP_WIDTH, MAP_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, sprite[s].texture);

            glBindVertexArray(sprite[s].VAO);
            glDrawArrays(GL_TRIANGLES, 0, 1 * 6);
          }

          snprintf(window_title, 255, "EPOCH %d/%d | ITERATION: %d/%d | RADIUS: %.2f | LEARNING RULE: %.2f\n", epoch, TOTAL_EPOCHS, iteration, iterations_per_epoch, radius, learning_rule);
          glfwSetWindowTitle(window, window_title);

          usleep(10000);
          glfwSwapBuffers(window);
  }

  for(int s=0; s<4; s++) {
    glDeleteVertexArrays(sprite[s].texture_id, &sprite[s].VAO);
    glDeleteBuffers(sprite[s].texture_id, &sprite[s].VBO);
    glDeleteBuffers(sprite[s].texture_id, &sprite[s].UBO);
    glDeleteTextures(sprite[s].texture_id, &sprite[s].texture_ref);
  }

  glfwTerminate();
  return 0;
}

void process_input(GLFWwindow *window)
{
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
        glViewport(0, 0, width, height);
}

void keyboard_callback(GLFWwindow* window, int key, int32 scancode, int32 action, int32 mode)
{
        switch(key) {
          case GLFW_KEY_ESCAPE: if(action == GLFW_PRESS) { glfwSetWindowShouldClose(window, GL_TRUE); } break;
        }
}
