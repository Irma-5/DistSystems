#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <mpi.h>
#include <mpi-ext.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N (2*2*2*2*2*2 + 2)
#define TAG_PASS_FIRST 100
#define TAG_PASS_LAST 200
#define N2 (N * N)
#define N3 (N * N2)
#define RESERVE_RANK (total_size - 1)


double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double w = 0.5;
double eps;
double b, s = 0.;
double A[N][N][N];

int relax1();
void init();
void verify();

void change_first();
void change_last();
int waitAll1();

int size, rank, left_idx, right_idx, step;
int total_size, total_rank, global_size, global_rank;
int rank_to_damage = 1;         
int iteration_to_damage = 50;
int fault_injected = 0;

MPI_Comm COMP_COMM = MPI_COMM_NULL;
MPI_Request req_buf[4];
MPI_Status stat_buf[4];

void error_handler(MPI_Comm *comm, int *err, ...) {
    MPIX_Comm_revoke(*comm);
}
void init_fault(int curr_iteration) {
    if (!fault_injected && rank == rank_to_damage && curr_iteration == iteration_to_damage) {
        printf("Процесс %d (global %d): имитация сбоя на итерации %d\n", 
               rank, global_rank, curr_iteration);
        fflush(stdout);
        raise(SIGKILL);
    }
}

void recompute_geometry() {
    MPI_Comm_size(COMP_COMM, &size);
    MPI_Comm_rank(COMP_COMM, &rank);
    int size_block = (N - 2) / size;
    int ost = (N - 2) % size;
    left_idx = 1;
    for (int i = 0; i < rank; i++) left_idx += (size_block + (i < ost ? 1 : 0));
    step = size_block + (rank < ost ? 1 : 0);
    right_idx = left_idx + step;
}

void save_checkpoint(int it) {
    MPI_File fh;
    MPI_Status status;
    MPI_Offset offset = sizeof(int) + (MPI_Offset)(left_idx - 1) * N2 * sizeof(double);
    if (MPI_File_open(COMP_COMM, "checkpoint.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh) == MPI_SUCCESS) {
        if (rank == 0) MPI_File_write_at(fh, 0, &it, 1, MPI_INT, &status);
        MPI_File_write_at_all(fh, offset, &A[left_idx][0][0], step * N2, MPI_DOUBLE, &status);
        MPI_File_close(&fh);
    }
}

int load_checkpoint() {
    MPI_File fh;
    MPI_Status status;
    int it = 0;
    if (MPI_File_open(COMP_COMM, "checkpoint.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) return 0;
    MPI_File_read_at(fh, 0, &it, 1, MPI_INT, &status);
    MPI_Offset offset = sizeof(int) + (MPI_Offset)(left_idx - 1) * N2 * sizeof(double);
    MPI_File_read_at_all(fh, offset, &A[left_idx][0][0], step * N2, MPI_DOUBLE, &status);
    MPI_File_close(&fh);
    return it;
}


int main(int an, char **as) {
    int it;

    MPI_Init(&an, &as);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    MPI_Comm_size(MPI_COMM_WORLD, &total_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &total_rank);

    int is_reserve = (total_rank == RESERVE_RANK);

    int color = is_reserve ? MPI_UNDEFINED : 0;
    MPI_Comm_split(MPI_COMM_WORLD, color, total_rank, &COMP_COMM);
    if (COMP_COMM != MPI_COMM_NULL) {
        MPI_Comm_set_errhandler(COMP_COMM, MPI_ERRORS_RETURN);
    }

    if (is_reserve) {
        int start_signal = 0;
        MPI_Status status;

        printf("Резервный процесс %d ожидает сигнала...\n", total_rank);
        fflush(stdout);

        MPIX_Comm_shrink(MPI_COMM_WORLD, &COMP_COMM);
        MPI_Comm_set_errhandler(COMP_COMM, MPI_ERRORS_RETURN);
        
        printf("Резервный процесс %d: подключился после сбоя\n", total_rank);
        fflush(stdout);


        recompute_geometry();
        fault_injected = 1;
        
        goto restart;
        // MPI_Recv(&start_signal, 1, MPI_INT, MPI_ANY_SOURCE, 999, MPI_COMM_WORLD, &status);
        if (start_signal == 1) {
            printf("Резервный процесс %d: получен сигнал начать работу!\n", total_rank);
            fflush(stdout);

            MPIX_Comm_shrink(MPI_COMM_WORLD, &COMP_COMM);
            MPI_Comm_set_errhandler(COMP_COMM, MPI_ERRORS_RETURN);

            MPI_Comm_size(COMP_COMM, &size);
            MPI_Comm_rank(COMP_COMM, &rank);

            recompute_geometry();

            goto restart;
        } else {
            MPI_Finalize();
            return 0;
        }
    } else {
        MPI_Comm_size(COMP_COMM, &size);
        MPI_Comm_rank(COMP_COMM, &rank);
    }

    int size_block = (N - 2) / size;
    int ost = (N - 2) % size;

    right_idx = 1;
    for (i = 0; i < rank + 1; i++) {
        right_idx += (size_block + (i < ost ? 1 : 0));
    }
    left_idx = right_idx - (size_block + (rank < ost ? 1 : 0));
    step = right_idx - left_idx;

    double time_start, time_end;
    time_start = MPI_Wtime();

    init();

restart:
    {
        int start_it = load_checkpoint();
        printf("Процесс %d начал считать с итерации %d\n", rank, start_it);

        for (it = start_it; it <= itmax; it++) {
            eps = 0.;

            init_fault(it);

            int rc = relax1();
            if (rc != MPI_SUCCESS) goto recover;

            if (it % 20 == 0) save_checkpoint(it);

            if (!rank)
                printf("it=%4i   eps=%f\n", it, eps);
            if (eps < maxeps)
                break;
        }
    }

    int *recv_counts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        recv_counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        int size_block = (N - 2) / size;
        int ost = (N - 2) % size;
        int current_displ = 0;
        
        for (int p = 0; p < size; p++) {
            int p_step = size_block + (p < ost ? 1 : 0);
            recv_counts[p] = p_step * N2;
            displs[p] = current_displ * N2;
            current_displ += p_step;
        }
    }

    MPI_Gatherv(&A[left_idx][0][0], step * N2, MPI_DOUBLE, 
                &A[1][0][0], recv_counts, displs, 
                MPI_DOUBLE, 0, COMP_COMM);

    if (rank == 0) {
        free(recv_counts);
        free(displs);
    }

    verify();
    time_end = MPI_Wtime();

    if (!rank)
        printf("Elapsed time: %lf.\n", time_end - time_start);

    MPI_Finalize();
    return 0;

recover:
    printf("Процесс %d: обнаружен сбой, восстановление...\n", total_rank);
    fflush(stdout);
    MPIX_Comm_revoke(COMP_COMM);
    MPI_Comm new_comm;
    MPIX_Comm_shrink(MPI_COMM_WORLD, &new_comm);
    
    if (COMP_COMM != MPI_COMM_NULL && COMP_COMM != MPI_COMM_WORLD) {
        MPI_Comm_free(&COMP_COMM);
    }
    COMP_COMM = new_comm;
    MPI_Comm_set_errhandler(COMP_COMM, MPI_ERRORS_RETURN);
    
    recompute_geometry();

    fault_injected = 1; 
    goto restart;
}

void init() {
    for (i = left_idx; i < right_idx; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++) {
                if (j == 0 || j == N - 1 || k == 0 || k == N - 1)
                    A[i][j][k] = 0.;
                else
                    A[i][j][k] = (4. + i + j + k);
            }
}

int relax1() {
    double eps_local = 0.;
    int rc;

    change_last();
    change_first();
    if (waitAll1() != MPI_SUCCESS) return -1;

    for (i = left_idx; i < right_idx; i++)
        for (j = 1; j <= N - 2; j++)
            for (k = 1 + (i + j) % 2; k <= N - 2; k += 2) {
                b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] +
                          A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
                eps_local = Max(fabs(b), eps_local);
                A[i][j][k] = A[i][j][k] + b;
            }

    change_last();
    change_first();
    if (waitAll1() != MPI_SUCCESS) return -1;

    for (i = left_idx; i < right_idx; i++)
        for (j = 1; j <= N - 2; j++)
            for (k = 1 + (i + j + 1) % 2; k <= N - 2; k += 2) {
                b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] +
                          A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
                A[i][j][k] = A[i][j][k] + b;
            }

    rc = MPI_Barrier(COMP_COMM);
    if (rc != MPI_SUCCESS) return -1;
    
    rc = MPI_Reduce(&eps_local, &eps, 1, MPI_DOUBLE, MPI_MAX, 0, COMP_COMM);
    if (rc != MPI_SUCCESS) return -1;
    
    rc = MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, COMP_COMM);
    if (rc != MPI_SUCCESS) return -1;

    return MPI_SUCCESS;
}

void verify() {
    double s_local = 0.;

    for (i = left_idx; i < right_idx; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++) {
                s_local = s_local + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N3);
            }
    MPI_Barrier(COMP_COMM);
    MPI_Reduce(&s_local, &s, 1, MPI_DOUBLE, MPI_SUM, 0, COMP_COMM);
    if (!rank) {
        printf("  S = %f\n", s);
    }
}


void change_last() {
    if (rank)
        MPI_Irecv(A[left_idx - 1], N2, MPI_DOUBLE, rank - 1, TAG_PASS_LAST, COMP_COMM, &req_buf[0]);
    if (rank != size - 1)
        MPI_Isend(A[right_idx - 1], N2, MPI_DOUBLE, rank + 1, TAG_PASS_LAST, COMP_COMM, &req_buf[2]);
}

void change_first() {
    if (rank != size - 1)
        MPI_Irecv(A[right_idx], N2, MPI_DOUBLE, rank + 1, TAG_PASS_FIRST, COMP_COMM, &req_buf[3]);
    if (rank)
        MPI_Isend(A[left_idx], N2, MPI_DOUBLE, rank - 1, TAG_PASS_FIRST, COMP_COMM,  &req_buf[1]);
}



int waitAll1() {
    int count = 4, shift = 0;
    if (!rank) {
        count -= 2;
        shift = 2;
    }
    if (rank == size - 1) {
        count -= 2;
    }
    
    if (count == 0) return MPI_SUCCESS;
    
    return MPI_Waitall(count, &req_buf[shift], &stat_buf[0]);
}
