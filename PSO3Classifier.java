package weka.classifiers.eas;


public class PSO3Classifier extends PSO1Classifier {



	public String globalInfo() {
		return "PSO con fitness 3 [De Falco, Della Cioppa, Tarantino, 2007].";
	}



	protected double evaluate(double[] x) {
		return fitness3(x);
	}



}

