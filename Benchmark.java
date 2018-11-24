import org.jblas.DoubleMatrix;
import org.jblas.NativeBlas;

import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.Random;

public class Benchmark {

// On my laptop, estimate 1.962 GFlops for DGEMM based on jblas.org benchmarks
private static final double MAX_SPEED = 1.962;
//DBL_EPSILON approximates the value defined in C's float.h header file
private static final double DBL_EPSILON = 222.0e-16;

//seed for testing purposes only--remove during actual runs
Random prng = new Random(20181122);

private void reference_dgemm (int N, double ALPHA, double[] A, double[] B, double[] C)
    {
        char TRANSA = 'N';
        char TRANSB = 'N';
        int M = N;
        int K = N;
        double BETA = 1.;
        int LDA = N;
        int LDB = N;
        int LDC = N;
        int aIdx = 0;
        int bIdx = 0;
        int cIdx =0;
        NativeBlas.dgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, aIdx, LDA, B, bIdx, LDB, BETA, C, cIdx, LDC);
    }

private double wall_time ()
    {
        return System.nanoTime()*10e-9;
    }

    private void die (String message)
    {
        System.out.printf (message);
        System.exit(-1);
    }

    /**
     * Fill this array with values uniformly distributed over [-1,1].
     * We do this element-by-element because we want to manage the allocated space
     * by hand.
     * @param p array to fill
     * @param n fill n x n patch of the array
     */
    private void fill (double[] p, int n)
    {
        for (int i = 0; i < n; ++i)
            p[i] = 2 * prng.nextDouble() - 1;
    }

    /**
     * Replace each negative value in the matrix with it's absolute value.
     * @param p the matrix to operate on
     * @param n the n x n patch of p to operate on
     */
    void absolute_value (double[] p, int n)
    {
        for (int i = 0; i < n; i++)
            p[i] = Math.abs (p[i]);
    }


    /* The benchmarking program */
    public static void main (String[] args)
    {
        Benchmark benchmark = new Benchmark();

        //load the algorithm from command line arg
        Class<?> taskClass = null;
        Dgemm task = null;
        String taskClassName = args[0];
        try {
            taskClass = Class.forName(taskClassName);
            if (Dgemm.class.isAssignableFrom(taskClass))
            {
                task = (Dgemm) taskClass.getDeclaredConstructor().newInstance();
            }
        } catch (ClassNotFoundException e) {
            usage(String.format("task class %s not found", taskClass));
        }
        catch (NoSuchMethodException e)
        {
            usage (String.format
                    ("Task class %s constructor cannot be accessed",
                            taskClass));
        }
        catch (IllegalAccessException | InvocationTargetException e)
        {
            usage (String.format
                    ("Task class %s or its nullary constructor cannot be accessed",
                            taskClass));
        }
        catch (InstantiationException e)
        {
            usage (String.format
                    ("Task class %s cannot be instantiated",
                            taskClass));
        }

        //begin the benchmark code
        System.out.printf ("Description:\t%s\n\n", task.getDgemmDesc());

        /* Test sizes should highlight performance dips at multiples of certain powers-of-two */

        int test_sizes[] =

                /* Multiples-of-32, +/- 1. Currently commented. */
                 {31,32,33,63,64,65,95,96,97,127,128,129,159,160,161,191,192,193,223,224,225,255,256,257,287,288,289,319,320,321,351,352,353,383,384,385,415,416,417,447,448,449,479,480,481,511,512,513,543,544,545,575,576,577,607,608,609,639,640,641,671,672,673,703,704,705,735,736,737,767,768,769,799,800,801,831,832,833,863,864,865,895,896,897,927,928,929,959,960,961,991,992,993,1023,1024,1025};

                 /* A representative subset of the first list. Currently uncommented.
                { 31, 32, 96, 97, 127, 128, 129, 191, 192, 229, 255, 256, 257,
                        319, 320, 321, 417, 479, 480, 511, 512, 639, 640, 767, 768, 769 };
                */

        int nsizes = test_sizes.length;

        /* assume last size is also the largest size */
        int nmax = test_sizes[nsizes-1];

        /* allocate memory for all problems */
        double[][] buf = null;
        try
        {
            buf = new double [3][nmax * nmax];
        }
        catch (OutOfMemoryError e) {
            benchmark.die ("failed to allocate largest problem size");
        }

        double Mflops_s[] = new double[nsizes];
        double per[] = new double[nsizes];
        double aveper;

        /* For each test size */
        for (int isize = 0; isize < nsizes; ++isize)
        {
            /* Create and fill 3 random matrices A,B,C*/
            int n = test_sizes[isize];

            double[] A = buf[0];
            double[] B = buf[1];
            double[] C = buf[2];

            benchmark.fill (A, n*n);
            benchmark.fill (B, n*n);
            benchmark.fill (C, n*n);

            /* Measure performance (in Gflops/s). */

            /* Time a "sufficiently long" sequence of calls to reduce noise */
            double Gflops_s=0;
            double seconds = -1.0;
            double timeout = 0.1; // "sufficiently long" := at least 1/10 second.
            for (int n_iterations = 1; seconds < timeout; n_iterations *= 2)
            {
                /* Warm-up */
                task.square_dgemm (n, A, B, C);

                /* Benchmark n_iterations runs of square_dgemm */
                seconds = -benchmark.wall_time();
                for (int it = 0; it < n_iterations; ++it)
                    task.square_dgemm (n, A, B, C);
                seconds += benchmark.wall_time();


                /*  compute Gflop/s rate */
                Gflops_s = 2.e-9 * n_iterations * n * n * n / seconds;
            }

            /* Storing Mflop rate and calculating percentage of peak */
            Mflops_s[isize] = Gflops_s*1000;
            per[isize] = Gflops_s*100/Benchmark.MAX_SPEED;

            System.out.printf ("Size: %d\tMflop/s: %8g\tPercentage:%6.2f\n", n, Mflops_s[isize],per[isize]);

            /* Ensure that error does not exceed the theoretical error bound. */

            /* C := A * B, computed with square_dgemm */
            Arrays.fill(C,0);
            task.square_dgemm (n, A, B, C);

            /* Do not explicitly check that A and B were unmodified on square_dgemm exit
             *  - if they were, the following will most likely detect it:
             * C := C - A * B, computed with reference_dgemm
             * should be near zero if user-defined dgemm routine called above is correct */
            benchmark.reference_dgemm(n, -1.0, A, B, C);

            /* A := |A|, B := |B|, C := |C| */
            benchmark.absolute_value (A, n * n);
            benchmark.absolute_value (B, n * n);
            benchmark.absolute_value (C, n * n);

            /* C := |C| - 3 * e_mach * n * |A| * |B|, computed with reference_dgemm */
            benchmark.reference_dgemm (n, -3.0*Benchmark.DBL_EPSILON*n, A, B, C);

            /* If any element in C is positive, then something went wrong in square_dgemm */
            for (int i = 0; i < n * n; ++i)
                if (C[i] > 0)
                    benchmark.die("*** FAILURE *** Error in matrix multiply exceeds componentwise error bounds.\n" );
        }

        /* Calculating average percentage of peak reached by algorithm */
        aveper=0;
        for (int i=0; i<nsizes;i++)
            aveper+= per[i];
        aveper/=nsizes*1.0;

        /* Printing average percentage to screen */
        System.out.printf("Average percentage of Peak = %g\n",aveper);
    }

    /**
     * Print an illegal argument usage message and exit.
     *
     * @param  arg  Command line argument.
     */
    private static void usageIllegal
    (String arg)
    {
        usage (arg + " illegal");
    }

    /**
     * Print a usage message and exit.
     *
     * @param  msg  Error message.
     */
    private static void usage
    (String msg)
    {
        System.err.printf ("benchmark: %s%n", msg);
        System.err.println ("Usage: java Benchmark <TaskClass>");
        System.exit (1);
    }

    public void print(double[] a, int n, String msg) {
        System.out.printf("\n%s\n", msg);
        for (int i = 0; i < n*n; i++) {
            if (i % n == 0) {System.out.printf("\n");}
            System.out.printf("%10.2f ", a[i]);
        }
        System.out.printf("\n");
    }
}
