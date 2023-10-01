
/*
 * java neural network for handwritten digit recognition written
 * to familiarize myself with the language for my MET CS342 class
 *
 * september 2023
 */

package Model;

import java.io.*;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;

public class Model {
    private Layer[] l;
    private double acc;
    private int classes;
    private int fc;
    private int sc;
    private Model[] mc;
    private LocalDateTime start;
    private LocalDateTime end;

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
                try {
                    mc[i].export("class"+i);
                } catch (IOException e) {
                    System.out.printf("model: I/O exception while exporting, ignoring\n");
                }
            }

            return true;
        }

        fc = x[0].length;
        sc = x.length;
        System.out.printf("model: training model on %d features\n", fc);

        start = LocalDateTime.now();

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

        /*try {
            export("test");
        } catch (IOException e) {
            System.out.printf("model: could not export model due to IO error; ignoring\n");
        }*/

        end = LocalDateTime.now();

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

    public void export(String name) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(name+".csv"));

        bw.write("name,"+name+"\n");
        bw.write("featureCount,"+fc+"\n");
        bw.write("sampleCount,"+sc+"\n");
        bw.write("layerCount,"+l.length+"\n");
        bw.write("accuracy,"+acc+"\n");

        DateTimeFormatter tf = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        bw.write("startTime,"+tf.format(start)+"\n");
        bw.write("endTime,"+tf.format(end)+"\n");

        // now iterate through layers
        int[] features;
        double[] weights;
        double bias;
        for(int i = 0; i < l.length; i++) {
            bw.write("layer"+i+"Activation," + l[i].activationFunction().toString() + "\n");
            bw.write("layer"+i+"Loss," + l[i].lossFunction().toString() + "\n");
            bw.write("layer"+i+"NeuronCount," + l[i].neuronCount() + "\n");

            // now loop through neurons
            for(int j = 0; j < l[i].neuronCount(); j++) {
                features = l[i].neuronFeatures(j);
                weights = l[i].neuron(j).weights();
                bias = l[i].neuron(j).bias();

                bw.write("layer"+i+"Neuron"+j+"Features,");
                for(int k = 0; k < features.length; k++) {
                    bw.write(""+features[k]);
                    if(k == (weights.length-1)) bw.write("\n");
                    else bw.write(",");
                }

                bw.write("layer"+i+"Neuron"+j+"Weights,");
                for(int k = 0; k < weights.length; k++) {
                    bw.write(""+weights[k]);
                    if(k == (weights.length-1)) bw.write("\n");
                    else bw.write(",");
                }

                bw.write("layer"+i+"Neuron"+j+"Bias,"+bias+"\n");
            }
        }

        bw.close();
    }
}
