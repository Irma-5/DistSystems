#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define GRID_SIZE 5
#define TAG_ROW 100
#define TAG_COL 200

int main(int argc, char **argv) {
    int rank, size;
    int my_row, my_col;
    double local_value, current_max, received_value;
    int buffer_size;
    void *buffer;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // будем считать что ранги выдаются последовательно (как будто матрица развернута в 1d)
    // исходя из этого считаем координаты

    my_row = rank / GRID_SIZE;
    my_col = rank % GRID_SIZE;
    
    local_value = (double)(((rank+1) * (my_row+1) * (my_col+1)) % 101) ;  // генерируем "случайное" значение в ячейке
    current_max = local_value;
    
    // printf("Процесс (%d,%d) rank=%d, значение=%.0f\n", my_row, my_col, rank, local_value);
    
    // создание буфера (данные + MPI_BSEND_OVERHEAD для каждого сбщ)
    buffer_size = (sizeof(double) + MPI_BSEND_OVERHEAD) * 2;
    buffer = malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);
    
    // фаза 1
    for (int step = GRID_SIZE - 1; step > 0; step--) {
        if (my_col == step) {
            int dest = my_row * GRID_SIZE + (my_col - 1);
            MPI_Bsend(&current_max, 1, MPI_DOUBLE, dest, TAG_ROW, MPI_COMM_WORLD);
        }
        
        if (my_col == step - 1) {
            int src = my_row * GRID_SIZE + (my_col + 1);
            MPI_Recv(&received_value, 1, MPI_DOUBLE, src, TAG_ROW, 
                     MPI_COMM_WORLD, &status);
            // бедем максимум текущего и полученного значений
            if (received_value > current_max) {
                current_max = received_value;
            }
        }
    }
    
    // фаза 2
    if (my_col == 0) {
        for (int step = GRID_SIZE - 1; step > 0; step--) {
            if (my_row == step) {
                int dest = (my_row - 1) * GRID_SIZE;
                MPI_Bsend(&current_max, 1, MPI_DOUBLE, dest, TAG_COL, MPI_COMM_WORLD);
            }
            
            if (my_row == step - 1) {
                int src = (my_row + 1) * GRID_SIZE;
                MPI_Recv(&received_value, 1, MPI_DOUBLE, src, TAG_COL, 
                         MPI_COMM_WORLD, &status); // тут можно не использовать IRecv т.к. блокировки не будет в силу топологии 
                if (received_value > current_max) {
                    current_max = received_value;
                }
            }
        }
    }
    
    // отелючение буфера
    int detach_size;
    void *detach_buffer;
    MPI_Buffer_detach(&detach_buffer, &detach_size);
    free(detach_buffer);
    
    if (rank == 0) {
        printf("результат:%.0f\n", current_max);
    }
    
    MPI_Finalize();
    return 0;
}