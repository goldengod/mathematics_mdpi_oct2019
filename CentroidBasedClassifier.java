package weka.classifiers.eas;

public class CentroidBasedClassifier extends EAClassifier {


	public String globalInfo() {
		return "Centroid Based [ROCCHIO] by Valentino.";
	}



	protected void makeCentroids() {
		//i -> classi
		//j -> attributi
		//k -> istanze
		//inizializzo valori centroidi e counters a 0
		int[] counters = new int[nc];
		for (int i=0; i<nc; i++)
			for (int j=0; j<na; j++) {
				centroids[i][j] = 0;
				counters[i] = 0;
			}
		//scandisco istanze
		for (int k=0; k<ni; k++) {
			//indice della classe dell'istanza corrente
			int ic = trainingClasses[k]-1; //e' 1-based
			//incremento counter ic
			counters[ic]++;
			//sommo attributi al rispettivo centroide
			for (int j=0; j<na; j++) {
				centroids[ic][j] += trainingInstances[k][j];
			}
		}
		//divido centroids per counters (se 0 rimetto centroids al centro dei maxminvalues)
		for (int i=0; i<nc; i++) {
			for (int j=0; j<na; j++) {
				if (counters[i]!=0)
					centroids[i][j] /= counters[i];
				else
					centroids[i][j] = (maxValues[j]-minValues[j])/2;
			}
		}
		//fine!!!
		//
		fitness1(centroids);
		vale_nfes = 49;
		fitness2(centroids);
		vale_nfes = 99;
		fitness3(centroids);
		vale_nfes = 149;
		fitness4(centroids);
		vale_nfes = 199;
		fitness5(centroids);
	}



}

