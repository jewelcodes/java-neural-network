
/*
 * java neural network for handwritten digit recognition written
 * to familiarize myself with the language for my MET CS342 class
 *
 * september 2023
 */

package Model;

public class Loss {
    public double logLoss(double[] p, double[] y) {
        double sum = 0;
        double sample;

        for(int i = 0; i < p.length; i++) {
            if(y[i] == 1) {
                sample = 1-Math.log10(p[i]);
            } else {
                sample = 1-Math.log10(1-p[i]);
            }
            sum += sample;
        }

        sum /= y.length;
        sum = 1-sum;

        return sum;
    }

    public double mseLoss(double[] p, double[] y) {
        double sum = 0;
        double sample;

        for(int i = 0; i < p.length; i++) {
            sample = y[i] - p[i];
            sample = Math.pow(sample, 2);
            sum += sample;
        }

        sum /= y.length;
        sum /= 2;
        return sum;
    }

    public double maeLoss(double[] p, double[] y) {
        double sum = 0;
        double sample;

        for(int i = 0; i < p.length; i++) {
            sample = y[i] - p[i];
            sum += sample;
        }

        sum /= y.length;
        return sum;
    }
}
