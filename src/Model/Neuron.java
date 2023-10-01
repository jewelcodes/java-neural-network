
/*
 * java neural network for handwritten digit recognition written
 * to familiarize myself with the language for my MET CS342 class
 *
 * september 2023
 */

package Model;

public class Neuron extends Activation {
    private ActivationFunction af;
    private LossFunction lf;
    private double[] w;
    private double b;
    private double learningRate = 0.1;
    private int epochs = 1100;
    private double[] a;
    private boolean off = false;

    private int nn, ln;

    public Neuron(ActivationFunction af_, LossFunction lf_) {
        af = af_;
        lf = lf_;
    }

    public Neuron clone() {
        Neuron n = new Neuron(af, lf);
        n.w = w;
        n.b = b;
        n.learningRate = learningRate;
        n.epochs = epochs;
        n.a = a;
        n.off = off;
        return n;
    }

    private double totalLoss(double[][] x, double[] y) {
        double[] p = new double[x.length];

        for(int i = 0; i < x.length; i++) {
            p[i] = predict(x[i]);
        }

        switch(lf) {
            case LOGLOSS:
                return logLoss(p, y);
            case MSE:
                return mseLoss(p, y);
            case MAE:
                return maeLoss(p, y);
            default:
                System.out.printf("UNDEFINED LOSS FUNCTION");
                return -1;
        }
    }

    public double accuracy(double[][] x, double[] y) {
        int correct = 0;
        int wrong = 0;
        double p;

        for(int i = 0; i < x.length; i++) {
            p = predict(x[i]);
            if(af == ActivationFunction.SIGMOID) {
                if (p >= 0.5) p = 1;
                else p = 0;
            } else {
                p = Math.round(p);
            }

            if(p == y[i]) {
                correct++;
            } else {
                wrong++;
            }
        }

        double a = (double)(correct*100)/(correct+wrong);
        return a;
    }

    public boolean train(double[][] x, double[] y, int nn_, int ln_) {
        System.out.printf("neuron %d:%d: training neuron on %d samples and %d features\n", ln_, nn_, x.length, x[0].length);
        ln = ln_;
        nn = nn_;

        w = new double[x[0].length];

        //System.out.printf("neuron: random starting w: ");
        for(int i = 0; i < x[0].length; i++) {
            w[i] = Math.random();
            //System.out.printf("%.3f ", w[i]);
        }

        //System.out.printf("\n");

        b = Math.random();
        //System.out.printf("neuron: random starting b: %.3f\n", b);

        double loss, devB;
        double[] devW = new double[x[0].length];
        double diff;

        double initialLoss = totalLoss(x, y);

        for(int i = 0; i < epochs; i++) {
            loss = totalLoss(x, y);

            if(i % 1500 == 0) {
                System.out.printf("neuron %d:%d: %d%% done (%d/%d), loss %.05f, accuracy %.1f%%\n", ln, nn, (i*100)/epochs, i, epochs, loss, accuracy(x, y));
            }

            // update parameters
            // FIXME: this is NOT mathematically correct for all activation/loss functions
            devB = 0;
            for(int j = 0; j < x[0].length; j++) {
                devW[j] = 0;
            }

            for(int j = 0; j < x.length; j++) { // iterate over SAMPLES
                diff = predict(x[j]) - y[j];
                devB += diff;

                for(int k = 0; k < x[0].length; k++) {  // iterate over FEATURES
                    devW[k] += (diff * x[j][k]);
                }
            }

            devB /= x.length;
            for(int k = 0; k < x[0].length; k++) {
                devW[k] /= x.length;
            }

            b -= (learningRate*devB);
            for(int k = 0; k < x[0].length; k++) {
                w[k] -= (learningRate*devW[k]);
            }
        }

        double lossChange = totalLoss(x, y) - initialLoss;
        /*if(lossChange > 0) {
            System.out.printf("neuron %d:%d: deeming selected features irrelevant\n", ln, nn);
            off = true;
        } else {*/
            System.out.printf("neuron %d:%d: training complete, loss %.05f, accuracy %.1f%%\n", ln, nn, totalLoss(x, y), accuracy(x, y));
            System.out.printf("neuron %d:%d final w: ", ln, nn);
            for (int i = 0; i < x[0].length; i++) {
                System.out.printf("%.3f ", w[i]);
            }

            System.out.printf("\n");
            System.out.printf("neuron %d:%d final b: %3f\n", ln, nn, b);
        /*}*/

        // list all activations
        a = new double[x.length];   // # samples
        for(int i = 0; i < x.length; i++) {
            a[i] = predict(x[i]);
        }

        return true;
    }

    public double predict(double[] x) {
        if(off) return 0;

        double sum = b;
        for(int i = 0; i < x.length; i++) {
            sum += w[i] * x[i];
        }

        switch(af) {
            case IDENTITY: return identity(sum);
            case SIGMOID: return sigmoid(sum);
            case RELU: return relu(sum);
            case LRELU: return lrelu(sum);
            case GAUSSIAN: return gaussian(sum);
            default:
                System.out.printf("UNDEFINED ACTIVATION FUNCTION\n");
                return 0;
        }
    }

    public double[] activations() {
        return a;
    }

    public double[] weights() {
        return w;
    }

    public double bias() {
        return b;
    }
}
