/*
Interface that facilitates the saving of the internal state of the classifier
 */
package tsc_algorithms;

/**
 *
 * @author ajb
 */
public interface SaveableEnsemble {
     public void saveResults(String tr, String te);
   
}
