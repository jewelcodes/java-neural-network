package Model;

public class Model {
    private Layer[] l;
    private double acc;
    private int classes;
    private Model[] mc;

    public Model(Layer[] l_) {
        l = l_;
        classes = 1;    // default binary classification
        System.out.printf("model: initialized with %d layers\n", l.length);
    }

    private boolean isInArray(double[] arr, double val) {
        for(int i = 0; i < arr.length; i++) {
            if(arr[i] == val) return true;
        }

        return false;
    }

    private int index(double[] arr, double val) {
        for(int i = 0; i < arr.length; i++) {
            if(arr[i] == val) return i;
        }

        return -1;
    }

    private double[][] classify(double[] y) {
        double[] uniqueValues = new double[classes];
        int u = 0;

        for(int i = 0; i < classes; i++) {
            uniqueValues[i] = Math.random();
        }

        for(int i = 0; i < y.length; i++) {
            if(!isInArray(uniqueValues, y[i])) {
                uniqueValues[u] = y[i];
                System.out.printf("model: assigning class value %d to raw value %.0f\n", u, y[i]);
                u++;
            }
        }

        double[][] classPred = new double[classes][];

        for(int i = 0; i < classes; i++) {
            classPred[i] = new double[y.length];

            for(int j = 0; j < y.length; j++) {
                if(index(uniqueValues, y[j]) == i) {
                    //System.out.printf("index %d is of class %d (%.0f)\n", j, i, uniqueValues[i]);
                    classPred[i][j] = 1;
                } else {
                    classPred[i][j] = 0;
                }
            }
        }

        return classPred;
    }

    public boolean train(double[][] x, double[] y) {
        // handle multiple classes
        if(classes > 1) {
            mc = new Model[classes];
            Layer[] lc;

            for(int i = 0; i < classes; i++) {
                lc = new Layer[l.length];
                for(int j = 0; j < l.length; j++) {
                    lc[j] = l[j].clone();
                }

                mc[i] = new Model(lc);
            }

            double[][] yc = classify(y);

            for(int i = 0; i < classes; i++) {
                System.out.printf("model: training for class %d\n", i);
                mc[i].train(x, yc[i]);
            }

            return true;
        }

        System.out.printf("model: training model on %d features\n", x[0].length);

        for(int i = 0; i < l.length; i++) {
            if (i == 0) {
                l[0].train(x, y, 0);
            } else {
                l[i].train(l[i - 1].a, y, i);
            }
        }

        acc = 0;
        int correct = 0;
        int wrong = 0;
        double p;

        // calculate accuracy over training set
        for(int i = 0; i < x.length; i++) {
            p = predict(x[i])[0];

            if (p >= 0.5) p = 1;
            else p = 0;

            if(p == y[i]) {
                correct++;
            } else {
                wrong++;
            }
        }

        acc = (double)(correct * 100)/(correct+wrong);
        System.out.printf("model: accuracy is %.1f%%\n", acc);

        return true;
    }

    public void setClasses(int c) {
        classes = c;
        System.out.printf("model: set classes to %d\n", c);
    }

    public double[] predict(double[] x) {
        if(classes > 1) {
            double[] p = new double[classes];
            for(int i = 0; i < classes; i++) {
                p[i] = mc[i].predict(x)[0];
                //System.out.printf("CLASS %d PROBABILITY: %.3f\n", i, p[i]);
            }

            return p;
        }

        // simply pass it through all the layers
        double[] a;
        a = l[0].predict(x);
        for(int i = 1; i < l.length; i++) {
            a = l[i].predict(a);
        }

        return a;
    }

    public double accuracy() {
        return acc;
    }
}
