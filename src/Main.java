
/*
 * java neural network for handwritten digit recognition written
 * to familiarize myself with the language for my MET CS342 class
 *
 * september 2023
 */

import Model.*;
import java.io.*;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        // load data
        int samples = 0;
        int features = 0;
        int line = 0;
        int position = 0;

        double[][] x = new double[1][1];
        double[] maxX = new double[1];
        double[] y = new double[1];

        try {
            Scanner s = new Scanner(new File("mnist_train.csv"));
            String st;
            String[] a;
            while(s.hasNextLine()) {
                st = s.nextLine();
                samples++;
            }

            samples--;

            s = new Scanner(new File("mnist_train.csv"));
            st = s.nextLine();
            a = st.split(",");
            features = a.length-1;
            System.out.printf("loaded %d samples with %d features\n", samples, features);

            x = new double[samples][features];
            y = new double[samples];
            maxX = new double[features];

            for(int i = 0; i < features; i++) {
                maxX[i] = 0;
            }

            line = 0;

            while(s.hasNextLine()) {
                st = s.nextLine();
                a = st.split(",");

                for(int i = 0; i < features; i++) {
                    x[line][i] = Double.parseDouble(a[i]);
                    if(x[line][i] > maxX[i]) {
                        maxX[i] = x[line][i];
                    }
                }

                y[line] = Double.parseDouble(a[features]);
                line++;
            }
        } catch(FileNotFoundException e) {
            System.out.printf("couldn't open file\n");
            System.exit((-1));
        }

        // regularize (?)
        for(int i = 0; i < x.length; i++) { // iterate over SAMPLES
            for(int j = 0; j < x[0].length; j++) {  // iterate over FEATURES
                x[i][j] /= maxX[j];
            }
        }

        // create a model
        Layer[] l = new Layer[3];
        l[0] = new Layer(400, ActivationFunction.RELU, LossFunction.MSE);
        l[1] = new Layer(50, ActivationFunction.RELU, LossFunction.MSE);
        l[2] = new Layer(1, ActivationFunction.SIGMOID, LossFunction.LOGLOSS);

        /*Layer[] l = new Layer[2];
        l[0] = new Layer(5, ActivationFunction.RELU, LossFunction.MSE);
        l[1] = new Layer(1, ActivationFunction.SIGMOID, LossFunction.LOGLOSS);*/

        // NOW TRAIN THE MODEL
        Model m = new Model(l);
        m.setClasses(10);
        m.train(x, y);

        // load test data
        double[][] xTest = new double[1][1];
        double[] yTest = new double[1];
        int testSamples = 0;

        try {
            Scanner s = new Scanner(new File("mnist_test.csv"));
            String st;
            String[] a;
            while(s.hasNextLine()) {
                st = s.nextLine();
                testSamples++;
            }

            testSamples--;

            s = new Scanner(new File("mnist_test.csv"));
            s.nextLine();

            xTest = new double[testSamples][features];
            yTest = new double[testSamples];

            line = 0;

            while(s.hasNextLine()) {
                st = s.nextLine();
                a = st.split(",");

                for(int i = 0; i < features; i++) {
                    xTest[line][i] = Double.parseDouble(a[i]);
                }

                yTest[line] = Double.parseDouble(a[features]);
                line++;
            }
        } catch(FileNotFoundException e) {
            System.out.printf("couldn't open file\n");
            System.exit((-1));
        }

        // regularize
        for(int i = 0; i < xTest.length; i++) { // iterate over SAMPLES
            for(int j = 0; j < xTest[0].length; j++) {  // iterate over FEATURES
                xTest[i][j] /= maxX[j];
            }
        }

        System.out.printf("loaded test data with %d samples\n", testSamples);

        // calculate accuracy over test dataset
        int correct = 0;
        int wrong = 0;
        double o[];

        for(int i = 0; i < xTest.length; i++) {
            o = m.predict(xTest[i]);
            if(argmax(o) != yTest[i]) {
                wrong++;
            } else {
                correct++;
            }
        }

        double accuracy = (double)(correct*100)/(correct+wrong);
        System.out.printf("accuracy over test dataset is %.1f%%\n", accuracy);

        // interactivity for no real reason
        Scanner si = new Scanner(System.in);
        String input;
        int entry;

        while(true) {
            System.out.printf("Enter an entry number to predict (or exit to quit): ");
            input = si.nextLine();
            if(input.equals("exit")) System.exit((0));
            else {
                entry = Integer.parseInt(input);
                if(entry >= xTest.length) {
                    System.out.printf("out of bounds; maximum is %d\n", xTest.length);
                } else {
                    o = m.predict(xTest[entry]);
                    System.out.printf("Expected %.0f, got %d [ ", yTest[entry], argmax(o));
                    for(int i = 0; i < o.length; i++) {
                        System.out.printf("%.2f ", o[i]);
                    }
                    System.out.printf("]\n");
                }
            }
        }
    }

    private static int argmax(double[] a) {
        int arg = 0;
        double max = -99999;
        for(int i = 0; i < a.length; i++) {
            if(a[i] > max) {
                max = a[i];
                arg = i;
            }
        }

        return arg;
    }
}