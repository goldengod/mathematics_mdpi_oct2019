package weka.classifiers.eas;


public class PSO4Classifier extends PSO1Classifier {



	public String globalInfo() {
		return "PSO con fitness 4 [De Falco, Della Cioppa, Tarantino, 2007].";
	}



	protected double evaluate(double[] x) {
		return fitness4(x);
	}



}

