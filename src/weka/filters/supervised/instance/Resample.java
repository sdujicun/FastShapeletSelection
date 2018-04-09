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
 *    Resample.java
 *    Copyright (C) 2002-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.filters.supervised.instance;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

/** 
 <!-- globalinfo-start -->
 * Produces a random subsample of a dataset using either sampling with replacement or without replacement.<br/>
 * The original dataset must fit entirely in memory. The number of instances in the generated dataset may be specified. The dataset must have a nominal class attribute. If not, use the unsupervised version. The filter can be made to maintain the class distribution in the subsample, or to bias the class distribution toward a uniform distribution. When used in batch mode (i.e. in the FilteredClassifier), subsequent batches are NOT resampled.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -S &lt;num&gt;
 *  Specify the random number seed (default 1)</pre>
 * 
 * <pre> -Z &lt;num&gt;
 *  The size of the output dataset, as a percentage of
 *  the input dataset (default 100)</pre>
 * 
 * <pre> -B &lt;num&gt;
 *  Bias factor towards uniform class distribution.
 *  0 = distribution in input data -- 1 = uniform distribution.
 *  (default 0)</pre>
 * 
 * <pre> -no-replacement
 *  Disables replacement of instances
 *  (default: with replacement)</pre>
 * 
 * <pre> -V
 *  Inverts the selection - only available with '-no-replacement'.</pre>
 * 
 <!-- options-end -->
 *
 * @author Len Trigg (len@reeltwo.com)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 8034 $ 
 */
public class Resample
  extends Filter 
  implements SupervisedFilter, OptionHandler {
  
  /** for serialization. */
  static final long serialVersionUID = 7079064953548300681L;

  /** The subsample size, percent of original set, default 100%. */
  protected double m_SampleSizePercent = 100;
  
  /** The random number generator seed. */
  protected int m_RandomSeed = 1;
  
  /** The degree of bias towards uniform (nominal) class distribution. */
  protected double m_BiasToUniformClass = 0;

  /** Whether to perform sampling with replacement or without. */
  protected boolean m_NoReplacement = false;

  /** Whether to invert the selection (only if instances are drawn WITHOUT 
   * replacement).
   * @see #m_NoReplacement */
  protected boolean m_InvertSelection = false;

  /**
   * Returns a string describing this filter.
   *
   * @return a description of the filter suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return 
        "Produces a random subsample of a dataset using either sampling "
      + "with replacement or without replacement.\n"
      + "The original dataset must "
      + "fit entirely in memory. The number of instances in the generated "
      + "dataset may be specified. The dataset must have a nominal class "
      + "attribute. If not, use the unsupervised version. The filter can be "
      + "made to maintain the class distribution in the subsample, or to bias "
      + "the class distribution toward a uniform distribution. When used in batch "
      + "mode (i.e. in the FilteredClassifier), subsequent batches are NOT resampled.";
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    Vector result = new Vector();

    result.addElement(new Option(
	"\tSpecify the random number seed (default 1)",
	"S", 1, "-S <num>"));

    result.addElement(new Option(
	"\tThe size of the output dataset, as a percentage of\n"
	+"\tthe input dataset (default 100)",
	"Z", 1, "-Z <num>"));

    result.addElement(new Option(
	"\tBias factor towards uniform class distribution.\n"
	+"\t0 = distribution in input data -- 1 = uniform distribution.\n"
	+"\t(default 0)",
	"B", 1, "-B <num>"));

    result.addElement(new Option(
	"\tDisables replacement of instances\n"
	+"\t(default: with replacement)",
	"no-replacement", 0, "-no-replacement"));

    result.addElement(new Option(
	"\tInverts the selection - only available with '-no-replacement'.",
	"V", 0, "-V"));

    return result.elements();
  }


  /**
   * Parses a given list of options. <p/>
   * 
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -S &lt;num&gt;
   *  Specify the random number seed (default 1)</pre>
   * 
   * <pre> -Z &lt;num&gt;
   *  The size of the output dataset, as a percentage of
   *  the input dataset (default 100)</pre>
   * 
   * <pre> -B &lt;num&gt;
   *  Bias factor towards uniform class distribution.
   *  0 = distribution in input data -- 1 = uniform distribution.
   *  (default 0)</pre>
   * 
   * <pre> -no-replacement
   *  Disables replacement of instances
   *  (default: with replacement)</pre>
   * 
   * <pre> -V
   *  Inverts the selection - only available with '-no-replacement'.</pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    String	tmpStr;
    
    tmpStr = Utils.getOption('S', options);
    if (tmpStr.length() != 0)
      setRandomSeed(Integer.parseInt(tmpStr));
    else
      setRandomSeed(1);

    tmpStr = Utils.getOption('B', options);
    if (tmpStr.length() != 0)
      setBiasToUniformClass(Double.parseDouble(tmpStr));
    else
      setBiasToUniformClass(0);

    tmpStr = Utils.getOption('Z', options);
    if (tmpStr.length() != 0)
      setSampleSizePercent(Double.parseDouble(tmpStr));
    else
      setSampleSizePercent(100);

    setNoReplacement(Utils.getFlag("no-replacement", options));

    if (getNoReplacement())
      setInvertSelection(Utils.getFlag('V', options));

    if (getInputFormat() != null) {
      setInputFormat(getInputFormat());
    }
  }

  /**
   * Gets the current settings of the filter.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {
    Vector<String>	result;

    result = new Vector<String>();

    result.add("-B");
    result.add("" + getBiasToUniformClass());

    result.add("-S");
    result.add("" + getRandomSeed());

    result.add("-Z");
    result.add("" + getSampleSizePercent());

    if (getNoReplacement()) {
      result.add("-no-replacement");
      if (getInvertSelection())
	result.add("-V");
    }
    
    return result.toArray(new String[result.size()]);
  }
    
  /**
   * Returns the tip text for this property.
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String biasToUniformClassTipText() {
    return "Whether to use bias towards a uniform class. A value of 0 leaves the class "
      + "distribution as-is, a value of 1 ensures the class distribution is "
      + "uniform in the output data.";
  }
    
  /**
   * Gets the bias towards a uniform class. A value of 0 leaves the class
   * distribution as-is, a value of 1 ensures the class distributions are
   * uniform in the output data.
   *
   * @return the current bias
   */
  public double getBiasToUniformClass() {
    return m_BiasToUniformClass;
  }
  
  /**
   * Sets the bias towards a uniform class. A value of 0 leaves the class
   * distribution as-is, a value of 1 ensures the class distributions are
   * uniform in the output data.
   *
   * @param newBiasToUniformClass the new bias value, between 0 and 1.
   */
  public void setBiasToUniformClass(double newBiasToUniformClass) {
    m_BiasToUniformClass = newBiasToUniformClass;
  }
    
  /**
   * Returns the tip text for this property.
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String randomSeedTipText() {
    return "Sets the random number seed for subsampling.";
  }
  
  /**
   * Gets the random number seed.
   *
   * @return the random number seed.
   */
  public int getRandomSeed() {
    return m_RandomSeed;
  }
  
  /**
   * Sets the random number seed.
   *
   * @param newSeed the new random number seed.
   */
  public void setRandomSeed(int newSeed) {
    m_RandomSeed = newSeed;
  }
    
  /**
   * Returns the tip text for this property.
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String sampleSizePercentTipText() {
    return "The subsample size as a percentage of the original set.";
  }
  
  /**
   * Gets the subsample size as a percentage of the original set.
   *
   * @return the subsample size
   */
  public double getSampleSizePercent() {
    return m_SampleSizePercent;
  }
  
  /**
   * Sets the size of the subsample, as a percentage of the original set.
   *
   * @param newSampleSizePercent the subsample set size, between 0 and 100.
   */
  public void setSampleSizePercent(double newSampleSizePercent) {
    m_SampleSizePercent = newSampleSizePercent;
  }
  
  /**
   * Returns the tip text for this property.
   * 
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String noReplacementTipText() {
    return "Disables the replacement of instances.";
  }

  /**
   * Gets whether instances are drawn with or without replacement.
   * 
   * @return true if the replacement is disabled
   */
  public boolean getNoReplacement() {
    return m_NoReplacement;
  }
  
  /**
   * Sets whether instances are drawn with or with out replacement.
   * 
   * @param value if true then the replacement of instances is disabled
   */
  public void setNoReplacement(boolean value) {
    m_NoReplacement = value;
  }
  
  /**
   * Returns the tip text for this property.
   * 
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String invertSelectionTipText() {
    return "Inverts the selection (only if instances are drawn WITHOUT replacement).";
  }

  /**
   * Gets whether selection is inverted (only if instances are drawn WIHTOUT 
   * replacement).
   * 
   * @return true if the replacement is disabled
   * @see #m_NoReplacement
   */
  public boolean getInvertSelection() {
    return m_InvertSelection;
  }
  
  /**
   * Sets whether the selection is inverted (only if instances are drawn WIHTOUT 
   * replacement).
   * 
   * @param value if true then selection is inverted
   */
  public void setInvertSelection(boolean value) {
    m_InvertSelection = value;
  }

  /** 
   * Returns the Capabilities of this filter.
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enableAllAttributes();
    result.enable(Capability.MISSING_VALUES);
    
    // class
    result.enable(Capability.NOMINAL_CLASS);
    
    return result;
  }
  
  /**
   * Sets the format of the input instances.
   *
   * @param instanceInfo an Instances object containing the input 
   * instance structure (any instances contained in the object are 
   * ignored - only the structure is required).
   * @return true if the outputFormat may be collected immediately
   * @throws Exception if the input format can't be set 
   * successfully
   */
  public boolean setInputFormat(Instances instanceInfo) 
       throws Exception {

    super.setInputFormat(instanceInfo);
    setOutputFormat(instanceInfo);
    return true;
  }

  /**
   * Input an instance for filtering. Filter requires all
   * training instances be read before producing output.
   *
   * @param instance the input instance
   * @return true if the filtered instance may now be
   * collected with output().
   * @throws IllegalStateException if no input structure has been defined
   */
  public boolean input(Instance instance) {

    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }
    if (m_NewBatch) {
      resetQueue();
      m_NewBatch = false;
    }
    if (isFirstBatchDone()) {
      push(instance);
      return true;
    } else {
      bufferInput(instance);
      return false;
    }
  }

  /**
   * Signify that this batch of input to the filter is finished. 
   * If the filter requires all instances prior to filtering,
   * output() may now be called to retrieve the filtered instances.
   *
   * @return true if there are instances pending output
   * @throws IllegalStateException if no input structure has been defined
   */
  public boolean batchFinished() {

    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }

    if (!isFirstBatchDone()) {
      // Do the subsample, and clear the input instances.
      createSubsample();
    }
    flushInput();

    m_NewBatch = true;
    m_FirstBatchDone = true;
    return (numPendingOutput() != 0);
  }

  /**
   * creates the subsample with replacement.
   * 
   * @param random		the random number generator to use
   * @param origSize		the original size of the dataset
   * @param sampleSize		the size to generate
   * @param actualClasses	the number of classes found in the data
   * @param classIndices	the indices where classes start
   */
  public void createSubsampleWithReplacement(Random random, int origSize, 
      int sampleSize, int actualClasses, int[] classIndices) {
    
    for (int i = 0; i < sampleSize; i++) {
      int index = 0;
      if (random.nextDouble() < m_BiasToUniformClass) {
	// Pick a random class (of those classes that actually appear)
	int cIndex = random.nextInt(actualClasses);
	for (int j = 0, k = 0; j < classIndices.length - 1; j++) {
	  if ((classIndices[j] != classIndices[j + 1]) && (k++ >= cIndex)) {
	    // Pick a random instance of the designated class
	    index =   classIndices[j] 
	            + random.nextInt(classIndices[j + 1] - classIndices[j]);
	    break;
	  }
	}
      }
      else {
	index = random.nextInt(origSize);
      }
      push((Instance) getInputFormat().instance(index).copy());
    }
  }

  /**
   * creates the subsample without replacement.
   * 
   * @param random		the random number generator to use
   * @param origSize		the original size of the dataset
   * @param sampleSize		the size to generate
   * @param actualClasses	the number of classes found in the data
   * @param classIndices	the indices where classes start
   */
  public void createSubsampleWithoutReplacement(Random random, int origSize, 
      int sampleSize, int actualClasses, int[] classIndices) {
    
    if (sampleSize > origSize) {
      sampleSize = origSize;
      System.err.println(
	  "Resampling without replacement can only use percentage <=100% - "
	  + "Using full dataset!");
    }

    Vector<Integer>[] indices = new Vector[classIndices.length - 1];
    Vector<Integer>[] indicesNew = new Vector[classIndices.length - 1];

    // generate list of all indices to draw from
    for (int i = 0; i < classIndices.length - 1; i++) {
      indices[i] = new Vector<Integer>(classIndices[i + 1] - classIndices[i]);
      indicesNew[i] = new Vector<Integer>(indices[i].capacity());
      for (int n = classIndices[i]; n < classIndices[i + 1]; n++)
	indices[i].add(n);
    }

    // draw X samples
    int currentSize = origSize;
    for (int i = 0; i < sampleSize; i++) {
      int index = 0;
      if (random.nextDouble() < m_BiasToUniformClass) {
	// Pick a random class (of those classes that actually appear)
	int cIndex = random.nextInt(actualClasses);
	for (int j = 0, k = 0; j < classIndices.length - 1; j++) {
	  if ((classIndices[j] != classIndices[j + 1]) && (k++ >= cIndex)) {
	    // no more indices for this class left, try again
	    if (indices[j].size() == 0) {
	      i--;
	      break;
	    }
	    // Pick a random instance of the designated class
	    index = random.nextInt(indices[j].size());
	    indicesNew[j].add(indices[j].get(index));
	    indices[j].remove(index);
	    break;
	  }
	}
      }
      else {
	index = random.nextInt(currentSize);
	for (int n = 0; n < actualClasses; n++) {
	  if (index < indices[n].size()) {
	    indicesNew[n].add(indices[n].get(index));
	    indices[n].remove(index);
	    break;
	  }
	  else {
	    index -= indices[n].size();
	  }
	}
	currentSize--;
      }
    }

    // sort indices
    if (getInvertSelection()) {
      indicesNew = indices;
    }
    else {
      for (int i = 0; i < indicesNew.length; i++)
	Collections.sort(indicesNew[i]);
    }

    // add to ouput
    for (int i = 0; i < indicesNew.length; i++) {
      for (int n = 0; n < indicesNew[i].size(); n++)
	push((Instance) getInputFormat().instance(indicesNew[i].get(n)).copy());
    }

    // clean up
    for (int i = 0; i < indices.length; i++) {
      indices[i].clear();
      indicesNew[i].clear();
    }
    indices = null;
    indicesNew = null;
  }

  /**
   * Creates a subsample of the current set of input instances. The output
   * instances are pushed onto the output queue for collection.
   */
  protected void createSubsample() {
    int origSize = getInputFormat().numInstances();
    int sampleSize = (int) (origSize * m_SampleSizePercent / 100);

    // Subsample that takes class distribution into consideration

    // Sort according to class attribute.
    getInputFormat().sort(getInputFormat().classIndex());
    
    // Create an index of where each class value starts
    int[] classIndices = new int [getInputFormat().numClasses() + 1];
    int currentClass = 0;
    classIndices[currentClass] = 0;
    for (int i = 0; i < getInputFormat().numInstances(); i++) {
      Instance current = getInputFormat().instance(i);
      if (current.classIsMissing()) {
	for (int j = currentClass + 1; j < classIndices.length; j++) {
	  classIndices[j] = i;
	}
	break;
      } else if (current.classValue() != currentClass) {
	for (int j = currentClass + 1; j <= current.classValue(); j++) {
	  classIndices[j] = i;
	}          
	currentClass = (int) current.classValue();
      }
    }
    if (currentClass <= getInputFormat().numClasses()) {
      for (int j = currentClass + 1; j < classIndices.length; j++) {
	classIndices[j] = getInputFormat().numInstances();
      }
    }
    
    int actualClasses = 0;
    for (int i = 0; i < classIndices.length - 1; i++) {
      if (classIndices[i] != classIndices[i + 1]) {
	actualClasses++;
      }
    }

    // Create the new sample
    Random random = new Random(m_RandomSeed);

    // Convert pending input instances
    if (getNoReplacement())
      createSubsampleWithoutReplacement(
	  random, origSize, sampleSize, actualClasses, classIndices);
    else
      createSubsampleWithReplacement(
	  random, origSize, sampleSize, actualClasses, classIndices);
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 8034 $");
  }
  
  /**
   * Main method for testing this class.
   *
   * @param argv should contain arguments to the filter: 
   * use -h for help
   */
  public static void main(String [] argv) {
    runFilter(new Resample(), argv);
  }
}
