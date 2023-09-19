package Model;

public class Layer {
    private Neuron[] n;
    private int nc;
    public double[][] a;
    private ActivationFunction af;
    private LossFunction lf;

    private int[][] features;   // which neurons have which features
                                // [neuron][0-FPN]

    public Layer(int c, ActivationFunction af_, LossFunction lf_) {
        nc = c;
        n = new Neuron[c];
        a = new double[c][];
        af = af_;
        lf = lf_;

        for(int i = 0; i < c; i++) {
            n[i] = new Neuron(af, lf);
        }

        System.out.printf("layer: initialized with %d neurons, %s activation and %s loss\n", c, af.toString(), lf.toString());
    }

    public Layer clone() {
        Layer l = new Layer(nc, af, lf);
        l.a = a;
        l.features = features;
        for(int i = 0; i < nc; i++) {
            l.n[i] = n[i].clone();
        }

        return l;
    }

    public ActivationFunction activationFunction() {
        return af;
    }

    private boolean isAllTaken(boolean[] xb) {
        for(int i = 0; i < xb.length; i++) {
            if(!xb[i]) return false;
        }

        return true;
    }

    private int firstUntaken(boolean[] xb) {
        for(int i = 0; i < xb.length; i++) {
            if(!xb[i]) return i;
        }

        return -1;
    }

    private void copyFeature(double[][] dst, int di, double[][] x, int f) {
        for(int i = 0; i < x.length; i++) {
            dst[i][di] = x[i][f];
        }
    }

    public boolean train(double[][] x, double[] y, int ln) {
        System.out.printf("layer %d: training %d neurons on %d features\n", ln, nc, x[0].length);

        a = new double[x.length][nc];

        // features per neuron
        int fpn;
        if(nc == x[0].length) {
            fpn = 1;
        } else if(nc != 1) {
            fpn = (x[0].length * 2) / nc;
            if (fpn <= 3) {
                fpn = 4;
            }
        } else {
            fpn = x[0].length;
        }

        System.out.printf("layer %d: %d features per neuron\n", ln, fpn);

        // randomize feature combinations (im not sure if this is correct)
        boolean[] xTaken = new boolean[x[0].length];   // whether we used this feature already
        for(int i = 0; i < x[0].length; i++) {
            xTaken[i] = false;
        }

        double[][] xn = new double[x.length][fpn];  // [samples][features]
        features = new int[nc][fpn];

        for(int i = 0; i < nc; i++) {
            System.out.printf("layer %d: setting up neuron %d\n", ln, i);
            // loop over neurons
            if(isAllTaken(xTaken)) {
                for(int j = 0; j < fpn; j++) {
                    int k = (int)Math.floor(Math.random()*x[0].length);

                    // copy feature
                    copyFeature(xn, j, x, k);
                    //System.out.printf("layer %d: assigned feature %d to neuron %d\n", ln, k, i);
                    features[i][j] = k;
                }
            } else {
                for(int j = 0; j < fpn; j++) {
                    int k = firstUntaken(xTaken);

                    if(k >= xTaken.length || k < 0) {
                        k = 0;
                    }

                    copyFeature(xn, j, x, k);
                    //System.out.printf("layer %d: assigned feature %d to neuron %d\n", ln, k, i);
                    features[i][j] = k;
                    xTaken[k] = true;
                    k++;
                }
            }

            n[i].train(xn, y, i, ln);

            // copy activations into an array of [sample][neuron] so the parent model can train the next layer
            for(int j = 0; j < x.length; j++) {
                a[j][i] = n[i].activations()[j];
            }
        }

        return true;
    }

    public double[] predict(double[] x) {
        // divide the features according to the diff neurons
        int fpn = features[0].length;
        double xn[][] = new double[nc][fpn];
        double output[] = new double[nc];

        for(int i = 0; i < nc; i++) {   // iterate over neurons
            for(int j = 0; j < fpn; j++) {  // iterate over features
                xn[i][j] = x[features[i][j]];
            }

            output[i] = n[i].predict(xn[i]);
        }

        return output;
    }
}
