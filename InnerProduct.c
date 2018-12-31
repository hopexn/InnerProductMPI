#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

//创建一个m x n 矩阵
float *create_matrix(int n, int m) {
    float *matrix;
    matrix = (float *) malloc(n * m * sizeof(float));
    return matrix;
}

//初始化矩阵
void init_matrix(float *matrix, int n, int m) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            matrix[i * m + j] = i + 1;
        }
    }
    return;
}

//求两个向量内积
float vector_product(float *row1, float *row2, int n) {
    int i;
    float result = 0;
    for (i = 0; i < n; i++) {
        result += row1[i] * row2[i];
    }
    return result;
}


int main(int argc, char **argv) {
    /**
     * num_procs： 处理器数量
     * my_rank: 当前进程的ID
     * prev_rank/next_rank: 上/下一个进程ID
     */
    int num_procs, my_rank, prev_rank, next_rank;
    /**
     * m：向量长度
     * n: 向量数量
     * rows_per_block: 分配给每个进程的向量数
     */
    int m, n, rows_per_block, i, j, pos;

    /**
     * mat: 存放数据的矩阵
     * sub_mat_u: 每个进程固定的向量块
     * sub_mat_v: 变化的向量块
     */
    float *mat = NULL, *sub_mat_u = NULL, *sub_mat_v = NULL;

    /**
     * statue* 发送、接收数据的状态
     * request* 发送、接收数据请求的状态
     * 两个变量用于阻塞
     */
    MPI_Status *statuses, status_u, status_v;
    MPI_Request *requests, request_u, request_v;

    /**
     * block_iter：
     * is_correct：
     */
    int block_iter, is_correct;

    /**
     * my_result: 当前进程的计算结果
     * final_result： 所有进程的计算结果，仅0号进程有效
     * serial_result：串行计算的结果，用于校验结果
     */
    float *my_result, *final_result, *serial_result;

    /**
     * 初始化MPI, 获取进程数以及当前进程ID
     */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    PMPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /**
     * m n：从参数中获取
     */
    m = atoi(argv[1]);
    n = atoi(argv[2]);

    /**
     * 获取相邻进程ID
     */
    prev_rank = (my_rank + num_procs - 1) % num_procs;
    next_rank = (my_rank + 1) % num_procs;

    /**
     * n个向量， num_procs个处理器， 将向量平均分给各个处理器
     */
    rows_per_block = n / num_procs;

    /**
     * 空间分配
     */
    sub_mat_u = create_matrix(rows_per_block, m);
    sub_mat_v = create_matrix(rows_per_block, m);
    requests = (MPI_Request *) malloc((num_procs - 1) * sizeof(MPI_Request));
    statuses = (MPI_Status *) malloc((num_procs - 1) * sizeof(MPI_Status));
    my_result = (float *) malloc(rows_per_block * n * sizeof(float));
    final_result = (float *) malloc(n * n * sizeof(float));
    serial_result = (float *) malloc(n * n * sizeof(float));
    memset(my_result, 0, rows_per_block * n * sizeof(float));

    /**
     * 主进程创建向量表，分块后将数据发给其余进程
     */
    if (my_rank == 0) {
        mat = create_matrix(n, m);
        init_matrix(mat, n, m);

        memcpy(sub_mat_u, mat, rows_per_block * m * sizeof(float));

        for (i = 1; i < num_procs; i++) {
            pos = i > my_rank ? i - 1 : i;
            MPI_Isend(mat + i * rows_per_block * m, rows_per_block * m, MPI_FLOAT, i, 0, MPI_COMM_WORLD,
                      requests + pos);
        }
        MPI_Waitall(num_procs - 1, requests, statuses);
    } else {
        /**
         * 接收从主线程来得数据
         */
        MPI_Irecv(sub_mat_u, rows_per_block * m, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request_u);
        MPI_Wait(&request_u, &status_u);
    }

    memcpy(sub_mat_v, sub_mat_u, rows_per_block * m * sizeof(float));
    /**
     * 每个进程将sub_mat_u发送到下一个进程，然后从上一个进程接收数据保存在sub_mat_u
     */
    for (block_iter = 0; block_iter < num_procs; block_iter++) {
        for (i = 0; i < rows_per_block; i++) {
            for (j = 0; j < rows_per_block; j++) {
                my_result[i * n + rows_per_block * ((my_rank + block_iter) % num_procs) + j] = vector_product(
                        sub_mat_u + i * m, sub_mat_v + j * m, m);
            }
        }
        MPI_Isend(sub_mat_v, rows_per_block * m, MPI_FLOAT, prev_rank, 0, MPI_COMM_WORLD, &request_v);
        MPI_Irecv(sub_mat_v, rows_per_block * m, MPI_FLOAT, next_rank, 0, MPI_COMM_WORLD, &request_v);
        MPI_Wait(&request_v, &status_v);
        MPI_Wait(&request_v, &status_v);
    }

    /**
     * 将结果归并
     * 主线程接收每个进程的数据， 其他进程则将数据发给主线程
     */
    if (my_rank == 0) {
        memcpy(final_result, my_result, rows_per_block * n * sizeof(float));
        for (block_iter = 1; block_iter < num_procs; block_iter++) {
            pos = block_iter - 1;
            MPI_Irecv(final_result + block_iter * rows_per_block * n, rows_per_block * n, MPI_FLOAT, block_iter, 0,
                      MPI_COMM_WORLD, requests + pos);
        }
        MPI_Waitall(num_procs - 1, requests, statuses);
    } else {
        MPI_Isend(my_result, rows_per_block * n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request_u);
        MPI_Wait(&request_u, &status_u);
    }

    /**
     * 显示、校验结果
     */
    if (my_rank == 0) {
        is_correct = 1;
        printf("Parallel result:\n");
        for (i = 0; i < n; i++) {
            for (j = i+1; j < n; j++) {
                printf("\t%.2f ", final_result[i * n + j]);
            }
            printf("\n");
        }
        printf("Serial result:\n");
        for (i = 0; i < n; i++) {
            for (j = i+1; j < n; j++) {
                serial_result[i * n + j] = vector_product(mat + i * m, mat + j * m, m);
                printf("\t%.2f ", serial_result[i * n + j]);
                if (serial_result[i * n + j] != final_result[i * n + j]) {
                    is_correct = 0;
                }
            }
            printf("\n");
        }
        if (is_correct == 1) {
            printf("Correct\n");
        } else {
            printf("Incorrect\n");
        }
        free(mat);
        free(final_result);
        free(serial_result);
    }
    /**
     * 释放资源
     */
    free(sub_mat_u);
    free(sub_mat_v);
    free(statuses);
    free(requests);
    free(my_result);
    MPI_Finalize();
    return 0;
}
