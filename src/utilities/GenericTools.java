/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities;

import java.util.ArrayList;

/**
 *
 * @author raj09hxu
 */
public class GenericTools {
    
    public static <E> ArrayList<E> cloneArrayList(ArrayList<E> list){
        ArrayList<E> temp = new ArrayList<>();
        for(E el : list){
           temp.add(el);
        }
        return temp;              
    }
}
