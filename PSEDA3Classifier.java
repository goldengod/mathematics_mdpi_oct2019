//VALENTINO 14 AGOSTO 2012
package weka.classifiers.eas;


public class PSEDA3Classifier extends PSEDA1Classifier {



	public String globalInfo() {
		return "PSEDA (global topology / variant 4) con wu=0.01 con fitness 3.";
	}



	protected double evaluate(double[] x) {
		return fitness3(x);
	}



}

