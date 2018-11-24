
public class DgemmNaive implements Dgemm {

    private final String dgemmDesc = "Naive, three-loop dgemm.";

    /* This routine performs a dgemm operation
     *  C := C + A * B
     * where A, B, and C are n x n matrices stored in column-major format.
     * On exit, A and B maintain their input values.
     * Column-major order is compatible with Fortran libraries.
     * This code doesn't use a native routine, but the benchmarking code does.
     */
    public void square_dgemm (int n, double[] A, double[] B, double[] C)
    {
        /* For each row i of A */
        for (int i = 0; i < n; i++) {
            /* For each column j of B */
            for (int j = 0; j < n; j++) {
                /* Compute C(i,j) */
                double cij = C[i+j*n];
                for( int k = 0; k < n; k++ )
                    cij += A[i+k*n] * B[k+j*n];
                C[i+j*n] = cij;
            }
        }
    }

    public String getDgemmDesc() {
        return dgemmDesc;
    }

}
