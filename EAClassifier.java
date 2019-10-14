package weka.classifiers.eas;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import java.util.Enumeration;
import weka.core.Attribute;
import java.util.Random;

import java.io.*;
import java.util.*;





public abstract class EAClassifier extends Classifier {



	public String globalInfo() {
		return "by VALENTINO.";
	}



	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
		// attributes
		//result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		// class
		result.enable(Capability.NOMINAL_CLASS);
		// instances
		result.setMinimumNumberInstances(0);
		return result;
	}



	//Training istances, classes, numero instances, classi e attributi (non classe) e anche minimi e massimi x attributo
	protected double[][] trainingInstances = null;
	protected int[] trainingClasses = null;
	protected int ni = 0;
	protected int nc = 0;
	protected int na = 0;
	protected double[] minValues = null;
	protected double[] maxValues = null;
	//I centroidi di tutte le classi
	protected double[][] centroids = null;
	//I classValues interni di weka
	protected double[] classValues = null;
	//Random number generator
	protected Random rng = null;



	public void buildClassifier(Instances instances) throws Exception {
		//can classifier handle the data?
		getCapabilities().testWithFail(instances);
		//remove instances with missing class
		instances = new Instances(instances);
		instances.deleteWithMissingClass();
		//controllo se l'attributo classe è l'ultimo
		if ( instances.classIndex() != instances.numAttributes()-1 )
			throw new Exception("VALENTINO: Attributo classe non e' l'ultimo!!!");
		//nome benchmark e' il nome dell'attributo classe!!!
		vale_benchmarkname = instances.classAttribute().name();
		//alloco trainingInstances e altro
		ni = instances.numInstances();
		nc = instances.numClasses();
		na = instances.numAttributes()-1;
		trainingInstances = new double[ni][na];
		trainingClasses = new int[ni];
		minValues = new double[na];
		maxValues = new double[na];
		classValues = new double[nc];
		//metto instances in trainingInstances/trainingClasses e salvo i classValues interni di weka e setto minValues/maxValues
		for (int j=0; j<na; j++) {
			minValues[j] = Double.POSITIVE_INFINITY;
			maxValues[j] = Double.NEGATIVE_INFINITY;
		}
		for (int i=0; i<ni; i++) {
			Instance instance = instances.instance(i);
			for (int j=0; j<na; j++) {
				Attribute attr = instance.attribute(j);
				if ( attr.isNominal() ) { //nominal
					trainingInstances[i][j] = (double)Integer.parseInt(instance.stringValue(j)); //perche' sono "1"..."X"
				} else { //numeric
					trainingInstances[i][j] = instance.value(j);
				}
				if ( trainingInstances[i][j]<minValues[j] )
					minValues[j] = trainingInstances[i][j];
				if ( trainingInstances[i][j]>maxValues[j] )
					maxValues[j] = trainingInstances[i][j];
			}
			int classLabel = Integer.parseInt(instance.stringValue(na)); //perche' sono "1"..."X"
			trainingClasses[i] = classLabel;
			classValues[classLabel-1] = instance.classValue(); // sono 1-based
		}
		//inizializzo generatore numeri random
		rng = new Random();
		//alloco centroids
		centroids = new double[nc][na];
		//creo centroidi
		makeCentroids();
		//disalloco trainingInstances - mantenere centroids e classValues e minValues e maxValues
		trainingInstances = null;
		trainingClasses = null;
		//disalloco rng
		rng = null;
		//chiudo file
		if (vale_output!=null) //nuovissimo
			vale_output.close();//nuovo
	}



	public double classifyInstance(Instance instance) throws Exception {
		//metto instance su array di double
		double[] inst = new double[na];
		for (int j=0; j<na; j++) {
			Attribute attr = instance.attribute(j);
			if ( attr.isNominal() ) { //nominal
				inst[j] = (double)Integer.parseInt(instance.stringValue(j)); //perche' sono "1"..."X"
			} else { //numeric
				inst[j] = instance.value(j);
			}
		}
		//calcolo distanza con tutti i centroidi e annoto la minima e il suo indice
		int imin = 0;
		double min = distance(inst,centroids[0]);
		for (int i=1; i<nc; i++) {
			double d = distance(inst,centroids[i]);
			if ( d<min ) {
				min = d;
				imin = i;
			}
		}
		//ritorno la classe con indice imin (e' 0-based)
		return classValues[imin];
	}



	protected abstract void makeCentroids();



	protected double distance(double[] a_orig, double[] b_orig) {
		//creo copie di a_orig e b_orig
		double[] a = new double[na];
		double[] b = new double[na];
		for (int j=0; j<na; j++) {
			a[j] = a_orig[j];
			b[j] = b_orig[j];
		}
		//calcolo gli array min e max x normalizzare
		double[] min = new double[na];
		double[] max = new double[na];
		for (int j=0; j<na; j++) {
			min[j] = minValues[j];
			if ( a[j]<min[j] )
				min[j] = a[j];
			if ( b[j]<min[j] )
				min[j] = b[j];
			max[j] = maxValues[j];
			if ( a[j]>max[j] )
				max[j] = a[j];
			if ( b[j]>max[j] )
				max[j] = b[j];
		}
		//normalizzo a e b
		for (int j=0; j<na; j++) {
			if ( max[j]!=min[j] ) {
				a[j] = (a[j] - min[j]) / (max[j] - min[j]);
				b[j] = (b[j] - min[j]) / (max[j] - min[j]);
			} else {
				a[j] = 0;
				b[j] = 0;
			}
		}
		//calcolo distanza euclidea fra a e b e la ritorno
		double result = 0;
		for (int j=0; j<na; j++) {
			result += (a[j] - b[j]) * (a[j] - b[j]);
		}
		result /= na; //come in [SALERNO] cosi' le distanze finali sono assicurate in [0,1]
		result = Math.sqrt(result); //potrebbe essere superfluo!!!
		return result;
	}



	protected double fitness1(double[][] x) { // x ha length nc*na
		//Inizializzo sommatoria
		int sum = 0;
		//per ogni training instance
		for (int i=0; i<ni; i++) {
			//calcolo il centroide con minima distanza
			int imin = 0;
			double min = distance(trainingInstances[i],x[0]);
			for (int j=1; j<nc; j++) {
				double d = distance(trainingInstances[i],x[j]);
				if ( d<min ) {
					min = d;
					imin = j;
				}
			}
			//incremento sommatoria se classe prevista e' errata (trainingClasses e' 1-based, imin e' 0-based)
			if ( trainingClasses[i]!=imin+1 )
				sum++;
		}
		//Moltiplico x 100 e divido x numero training instances e ritorno
		double result = ( sum * 100.0 ) / ni;
		vale_print(result);//nuovo
		return result;
	}



	protected double fitness2(double[][] x) {
		//Inizializzo sommatoria
		double sum = 0;
		//per ogni training instance
		for (int i=0; i<ni; i++) {
			//incremento sommatoria con distanza fra training istance e centroide della classe a cui appartiene (trainingClasses e' 1-based, x e' 0-based)
			sum += distance(trainingInstances[i],x[trainingClasses[i]-1]);
		}
		//Divido sum x numero training instances e ritorno
		double result = sum / ni;
		vale_print(result);//nuovo
		return result;
	}



	protected double innerFitness3(double[][] x,double w1,double w2) {
		//Inizializzo sommatoria
		int sum1 = 0;
		double sum2 = 0;
		//per ogni training instance
		for (int i=0; i<ni; i++) {
			//calcolo il centroide con minima distanza
			int imin = 0;
			double min = distance(trainingInstances[i],x[0]);
			if (0==trainingClasses[i]-1)
				sum2 += min;
			for (int j=1; j<nc; j++) {
				double d = distance(trainingInstances[i],x[j]);
				if ( d<min ) {
					min = d;
					imin = j;
				}
				if (j==trainingClasses[i]-1)
					sum2 += d;
			}
			//incremento sommatoria se classe prevista e' errata (trainingClasses e' 1-based, imin e' 0-based)
			if ( trainingClasses[i]!=imin+1 )
				sum1++;
		}
		//Moltiplico x 100 e divido x numero training instances e ritorno
		//double result = ( sum * 100.0 ) / ni;
		double result1 = ((double)sum1)/((double)ni);
		double result2 = sum2/ni;
		return w1*result1+w2*result2;
	}



	protected double fitness3(double[][] x) {
		/*double f1 = fitness1(x);
		double f2 = fitness2(x);
		double result = ( f1/100.0 + f2 ) / 2.0;
		vale_print(result);//nuovo
		return result;*/
		double result = innerFitness3(x,0.5,0.5);
		vale_print(result);
		return result;
	}



	protected double fitness4(double[][] x) {
		double result = innerFitness3(x,0.25,0.75);
		vale_print(result);
		return result;
	}



	protected double fitness5(double[][] x) {
		double result = innerFitness3(x,0.75,0.25);
		vale_print(result);
		return result;
	}



	protected double fitness1(double[] x) {
		copyArraySolutionToMatrix(x,centroids);
		return fitness1(centroids);
	}



	protected double fitness2(double[] x) {
		copyArraySolutionToMatrix(x,centroids);
		return fitness2(centroids);		
	}



	protected double fitness3(double[] x) {
		copyArraySolutionToMatrix(x,centroids);
		return fitness3(centroids);		
	}



	protected double fitness4(double[] x) {
		copyArraySolutionToMatrix(x,centroids);
		return fitness4(centroids);		
	}



	protected double fitness5(double[] x) {
		copyArraySolutionToMatrix(x,centroids);
		return fitness5(centroids);		
	}



	protected void copyArraySolutionToMatrix(double[] arr, double[][] mat) { //arr ha length nc*na
		int k = 0;
		for (int i=0; i<nc; i++) {
			for (int j=0; j<na; j++) {
				mat[i][j] = arr[k];
				k++;
			}
		}
	}



	//19 ottobre 2012
	protected int vale_nfes = 0;
	protected transient PrintStream vale_output = null;
	protected double vale_fbest = Double.POSITIVE_INFINITY;
	protected String vale_benchmarkname;



	protected void vale_print(double f) {
		if (vale_nfes==0) {
			String classname = this.getClass().getName();
			String vale_filename = "/ev_nfs_shared/output_weka/"+classname.substring(classname.lastIndexOf(".")+1)+"_"+vale_benchmarkname+"_"+Math.abs(rng.nextLong())+".txt";
			//System.out.println(vale_filename);
			try {
				vale_output = new PrintStream(new FileOutputStream(vale_filename));
			} catch (Exception e) {
				System.err.println("VALENTINO!!!");
				e.printStackTrace();
			}
			vale_output.print("0,");
			vale_output.printf(Locale.US,"%.8f",f);
			vale_output.println();
		}
		vale_nfes++;
		if (f<vale_fbest)
			vale_fbest = f;
		if (vale_nfes%50==0) {
			vale_output.printf(Locale.US,"%d",vale_nfes);
			vale_output.print(",");
			vale_output.printf(Locale.US,"%.8f",vale_fbest);
			vale_output.println();
		}
	}


}

