#ifndef __RAYTRACING_H
#define __RAYTRACING_H

#include "objects.h"
#include <stdint.h>


#ifdef MULTI_THREAD_CYCLE_PARTITION

typedef struct {
    point3 u;
    point3 v;
    point3 w;
    uint8_t trd_num;
    int w_idx;
    int h_idx;
    int factor;
    int width;
    int height;
    uint8_t *pixels;
    const viewpoint *view;
    sphere_node *p_spheres;
    light_node *p_lights;
    rectangular_node *p_rectangulars;
    color bg_color;
    uint32_t total_num;
    uint8_t thread_num;
    uint8_t start_idx;
} thread_param_t;

#else

typedef struct {
    point3 u;
    point3 v;
    point3 w;
    uint8_t trd_num;
    int w_idx;
    int h_idx;
    int factor;
    int width;
    int height;
    uint8_t *pixels;
    const viewpoint *view;
    sphere_node *p_spheres;
    light_node *p_lights;
    rectangular_node *p_rectangulars;
    color bg_color;
    uint32_t total_num;
} thread_param_t;


#endif
void raytracing(uint8_t *pixels, color background_color,
                rectangular_node rectangulars, sphere_node spheres,
                light_node lights, const viewpoint *view,
                int width, int height);
#endif
