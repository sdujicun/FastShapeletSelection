/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tsc_algorithms;

import java.io.Serializable;

/**
 *
 * @author ajb
 */
public interface ParameterSplittable extends Serializable{
    public void setParamSearch(boolean b);
/* The actual parameter values should be set internally. This integer
  is just a key to maintain different parameter sets. The range starts at 1
    */
    public void setPara(int x);
    public String getParas();
    double getAcc();    
}
