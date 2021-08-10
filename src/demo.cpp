/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: 774074168@qq.com
 */

#include <iostream>
#include <functional>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1


void get_input_fp32_data(const char* image_file, float* input_data, int letterbox_rows, int letterbox_cols, const float* mean, const float* scale)
{
    cv::Mat sample = cv::imread(image_file, 1);
    cv::Mat img;

    if (sample.channels() == 1)
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);

    cv::resize(img, img, cv::Size(letterbox_cols, letterbox_rows));

    cv::Mat img_new(letterbox_cols, letterbox_rows, CV_8UC3,cv::Scalar(0,0,0));
    img.convertTo(img_new, CV_32FC3);
    float* img_data   = (float* )img_new.data;

    /* nhwc to nchw */
    for (int h = 0; h < letterbox_rows; h++)
    {
        for (int w = 0; w < letterbox_cols; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index  = h * letterbox_cols * 3 + w * 3 + c;
                int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                input_data[out_index] = (img_data[in_index] - mean[c]) * scale[c];
            }
        }
    }

    
}
void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char* model_file = nullptr;
    char* image_file = nullptr;
    int img_h = 512;
    int img_w = 1024;
    const float mean[3] = { 123.675f, 116.28f,  103.53f };
    const float scale[3] = { 0.01712475f, 0.0175f, 0.01742919f };

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:")) != -1)
    {
        switch (res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            case 'r':
                repeat_count = atoi(optarg);
                break;
            case 't':
                num_thread = atoi(optarg);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    /* check files */
    if (model_file == nullptr)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (image_file == nullptr)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    init_tengine();
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3,img_h, img_w};    // nchw
    float* input_data = (float* )malloc(img_size * sizeof(float));

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data, img_size * sizeof(float)) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    get_input_fp32_data(image_file, input_data, img_h, img_w, mean, scale);
    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int i = 0; i < repeat_count; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        if (min_time > cur)
            min_time = cur;
        if (max_time < cur)
            max_time = cur;
    }
    printf("Repeat [%d] min %.3f ms, max %.3f ms, avg %.3f ms\n", repeat_count, min_time, max_time,
           total_time / repeat_count);

    /* get output tensor */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);

    float* data = ( float* )(get_tensor_buffer(output_tensor));

    int cityscapes_palette[][3]={{128, 64, 128},{244, 35, 232},{70, 70, 70},{102, 102, 156},
            {190, 153, 153},{153, 153, 153},{250, 170, 30},{220, 220, 0},
            {107, 142, 35},{152, 251, 152},{70, 130, 180},{220, 20, 60},
            {255, 0, 0},{0, 0, 142},{0, 0, 70},{0, 60, 100},{0, 80, 100},
            {0, 0, 230},{119, 11, 32}};


    int ade_palette[][3]={{120, 120, 120},{180, 120, 120},{6, 230, 230},{80, 50, 50},
            {4, 200, 3},{120, 120, 80},{140, 140, 140},{204, 5, 255},
            {230, 230, 230},{4, 250, 7},{224, 5, 255},{235, 255, 7},
            {150, 5, 61},{120, 120, 70},{8, 255, 51},{255, 6, 82},
            {143, 255, 140},{204, 255, 4},{255, 51, 7},{204, 70, 3},
            {0, 102, 200},{61, 230, 250},{255, 6, 51},{11, 102, 255},
            {255, 7, 71},{255, 9, 224},{9, 7, 230},{220, 220, 220},
            {255, 9, 92},{112, 9, 255},{8, 255, 214},{7, 255, 224},
            {255, 184, 6},{10, 255, 71},{255, 41, 10},{7, 255, 255},
            {224, 255, 8},{102, 8, 255},{255, 61, 6},{255, 194, 7},
            {255, 122, 8},{0, 255, 20},{255, 8, 41},{255, 5, 153},
            {6, 51, 255},{235, 12, 255},{160, 150, 20},{0, 163, 255},
            {140, 140, 140},{250, 10, 15},{20, 255, 0},{31, 255, 0},
            {255, 31, 0},{255, 224, 0},{153, 255, 0},{0, 0, 255},
            {255, 71, 0},{0, 235, 255},{0, 173, 255},{31, 0, 255},
            {11, 200, 200},{255, 82, 0},{0, 255, 245},{0, 61, 255},
            {0, 255, 112},{0, 255, 133},{255, 0, 0},{255, 163, 0},
            {255, 102, 0},{194, 255, 0},{0, 143, 255},{51, 255, 0},
            {0, 82, 255},{0, 255, 41},{0, 255, 173},{10, 0, 255},
            {173, 255, 0},{0, 255, 153},{255, 92, 0},{255, 0, 255},
            {255, 0, 245},{255, 0, 102},{255, 173, 0},{255, 0, 20},
            {255, 184, 184},{0, 31, 255},{0, 255, 61},{0, 71, 255},
            {255, 0, 204},{0, 255, 194},{0, 255, 82},{0, 10, 255},
            {0, 112, 255},{51, 0, 255},{0, 194, 255},{0, 122, 255},
            {0, 255, 163},{255, 153, 0},{0, 255, 10},{255, 112, 0},
            {143, 255, 0},{82, 0, 255},{163, 255, 0},{255, 235, 0},
            {8, 184, 170},{133, 0, 255},{0, 255, 92},{184, 0, 255},
            {255, 0, 31},{0, 184, 255},{0, 214, 255},{255, 0, 112},
            {92, 255, 0},{0, 224, 255},{112, 224, 255},{70, 184, 160},
            {163, 0, 255},{153, 0, 255},{71, 255, 0},{255, 0, 163},
            {255, 204, 0},{255, 0, 143},{0, 255, 235},{133, 255, 0},
            {255, 0, 235},{245, 0, 255},{255, 0, 122},{255, 245, 0},
            {10, 190, 212},{214, 255, 0},{0, 204, 255},{20, 0, 255},
            {255, 255, 0},{0, 153, 255},{0, 41, 255},{0, 255, 204},
            {41, 0, 255},{41, 255, 0},{173, 0, 255},{0, 245, 255},
            {71, 0, 255},{122, 0, 255},{0, 255, 184},{0, 92, 255},
            {184, 255, 0},{0, 133, 255},{255, 214, 0},{25, 194, 194},
            {102, 255, 0},{92, 0, 255}};

    //argmax
    int target_size_h = img_h;
    int target_size_w = img_w;
    cv::Mat segidx = cv::Mat::zeros(target_size_h, target_size_w, CV_8UC1);
    unsigned char* segidx_data = segidx.data;
    int h = target_size_h;
    int w = target_size_w;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            int maxk = 0;
            float tmp = data[0 * w * h + i * w + j];
            for (int k = 0; k < 19; k++)//ade-150,city-19
            {
                if (tmp < data[k * w * h + i * w + j])
                {
                    tmp = data[k * w * h + i * w + j];
                    maxk = k;
                }

            }

            segidx_data[i * w + j] = maxk;
        }
    }

    //draw result
    cv::Mat src = cv::imread(image_file);
    cv::Mat maskResize;
    cv::resize(segidx, maskResize, cv::Size(src.cols, src.rows), 0, 0, cv::INTER_NEAREST);
    for (size_t h = 0; h < src.rows; h++)
    {
        cv::Vec3b* pRgb = src.ptr<cv::Vec3b >(h);
        for (size_t w = 0; w < src.cols; w++)
        {
            int index = maskResize.at<uchar>(h, w);
            pRgb[w] = cv::Vec3b(cityscapes_palette[index][2] * 0.6 + pRgb[w][2] * 0.4, cityscapes_palette[index][1] * 0.6 + pRgb[w][1] * 0.4, cityscapes_palette[index][0] * 0.6 + pRgb[w][0] * 0.4);
        }
    }
    cv::imwrite("segformer_result.jpg",src);
    //cv::imshow("result",src);
    //cv::waitKey();


    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
