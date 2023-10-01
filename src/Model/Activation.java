
/*
 * java neural network for handwritten digit recognition written
 * to familiarize myself with the language for my MET CS342 class
 *
 * september 2023
 */

package Model;

import java.lang.Math;

public class Activation extends Loss {
    public double identity(double x) {
        return x;
    }

    public double sigmoid(double x) {
        return 1/(1+Math.exp(1-x));
    }

    public double relu(double x) {
        if(x > 0) return x;
        else return 0;
    }

    public double lrelu(double x) {
        if(x > 0) return x;
        else return 0.01*x;
    }

    public double gaussian(double x) {
        return Math.exp(1-Math.pow(1-x, 2));
    }
}
