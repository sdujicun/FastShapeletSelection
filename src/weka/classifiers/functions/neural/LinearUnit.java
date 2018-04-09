/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    LinearUnit.java
 *    Copyright (C) 2001-2012 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.functions.neural;

import weka.core.RevisionHandler;
import weka.core.RevisionUtils;

/**
 * This can be used by the 
 * neuralnode to perform all it's computations (as a Linear unit).
 *
 * @author Malcolm Ware (mfw4@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public class LinearUnit
  implements NeuralMethod, RevisionHandler {

  /** for serialization */
  private static final long serialVersionUID = 8572152807755673630L;
  
  /**
   * This function calculates what the output value should be.
   * @param node The node to calculate the value for.
   * @return The value.
   */
  public double outputValue(NeuralNode node) {
    double[] weights = node.getWeights();
    NeuralConnection[] inputs = node.getInputs();
    double value = weights[0];
    for (int noa = 0; noa < node.getNumInputs(); noa++) {
      
      value += inputs[noa].outputValue(true) 
	* weights[noa+1];
    }
     
    return value;
  }
  
  /**
   * This function calculates what the error value should be.
   * @param node The node to calculate the error for.
   * @return The error.
   */
  public double errorValue(NeuralNode node) {
    //then calculate the error.
    
    NeuralConnection[] outputs = node.getOutputs();
    int[] oNums = node.getOutputNums();
    double error = 0;
 
    for (int noa = 0; noa < node.getNumOutputs(); noa++) {
      error += outputs[noa].errorValue(true) 
	* outputs[noa].weightValue(oNums[noa]);
    }
    return error;
  }

  /**
   * This function will calculate what the change in weights should be
   * and also update them.
   * @param node The node to update the weights for.
   * @param learn The learning rate to use.
   * @param momentum The momentum to use.
   */
  public void updateWeights(NeuralNode node, double learn, double momentum) {

    NeuralConnection[] inputs = node.getInputs();
    double[] cWeights = node.getChangeInWeights();
    double[] weights = node.getWeights();
    
    double learnTimesError = 0;
    learnTimesError = learn * node.errorValue(false);
    
    double c = learnTimesError + momentum * cWeights[0];
    weights[0] += c;
    cWeights[0] = c;
      
    int stopValue = node.getNumInputs() + 1;
    for (int noa = 1; noa < stopValue; noa++) {
      
      c = learnTimesError * inputs[noa-1].outputValue(false);
      c += momentum * cWeights[noa];
      
      weights[noa] += c;
      cWeights[noa] = c; 
    }
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 8034 $");
  }
}
