/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package applications;

import examples.TransformExamples;
import fileIO.OutFile;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import utilities.ClassifierTools;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Instance;
import weka.filters.*;

/**
 *
 * @author RooTsMan v4.0 n.a.i
 */
public class ElectricityProfileClassification {

    //static String loaddummypath = "C:\\Users\\RooTsMan v4.0 n.a.i\\Documents\\UEA\\Year 4\\MCOMP project\\experiments\\ElectricDevices\\";
    //static String loadcountypath = "C:\\Users\\RooTsMan v4.0 n.a.i\\Documents\\UEA\\Year 4\\MCOMP project\\experiments\\DataForPauwe\\DataForPauwe\\county\\";
    static String loadpath = "C:\\Users\\RooTsMan v4.0 n.a.i\\Documents\\UEA\\Year 4\\MCOMP project\\experiments\\DataForPauwe\\DataForPauwe\\";
    static String storepath = "C:\\Users\\RooTsMan v4.0 n.a.i\\Documents\\UEA\\Year 4\\MCOMP project\\experiments\\results\\";
    static String county = "county";
    static String householdIncome = "householdIncome";
    static String propertyType = "propertyType";
    static String yearBuilt = "yearBuilt";
    //static String county = "county\\";
    //static String householdIncome = "householdIncome\\";
    //static String propertyType = "propertyType\\";
    //static String yearBuilt = "yearBuilt\\";
    
    public static String[] CountyFiles = {
        "county_DUABAAAL",
        "county_DUABAAAY",
        "county_DUABAABH",
        "county_DUABAABK",
        "county_DUABAABO",
        "county_DUABAABR",
        "county_DUABAABV",
        "county_DUABAABW",
        "county_DUABAABY",
        "county_DUABAACA",
        "county_DUABAACH",
        "county_duabaaci",
        "county_DUABAACK",
        "county_DUABAACS",
        "county_DUABAACX",
        "county_DUACAAAE",
        "county_duacaaaq",
        "county_duacaaas",
        "county_duacaahc",
        "county_SOACAABI",
        "county_SOACAABL",
        "county_SOACAABM",
        "county_SOACAACL",
        "county_SOACAAEF",
        "county_SOACAAEL",
        "county_soacaafi",
        "county_SOACAAGH",
        "county_SOACAAGI",
        "county_SOADAAAY",
        "county_SOADAABB",
        "county_SOADAABF",
        "county_SOADAABI",
        "county_soadaabr",
        "county_SOADAALV",
        "county_TRI-PPB-PETTY",
        "county_TRI-VET-AAAAA",
        "county_TRI-VET-AAAAB",
        "county_TRI-VET-AAAAC",
        "county_TRI-VET-AAAAD",
        "county_TRI-VET-AAAAE",
        "county_TRI-VET-AAAAG",
        "county_TRI-VET-AAAAJ",
        "county_TRI-VET-AAAAK",
        "county_TRI-VET-AAAAL",
        "county_TRI-VET-AAAAN",
        "county_TRI-VET-AAAAO",
        "county_TRI-VET-AAAAP",
        "county_TRI-VET-AAAAR",
        "county_TRI-VET-AAAAS",
        "county_TRI-VET-AAAAU",
        "county_TRI-VET-AAAAX",
        "county_TRI-VET-AAAAZ",
        "county_TRI-VET-AAABC",
        "county_TRI-VET-AAABD",
        "County_TRI-VET-AAABF",
        "county_TRI-VET-AAABH",
        "county_TRI-VET-AAABJ",
        "county_TRI-VET-AAABK",
        "county_TRI-VET-AAABM",
        "county_TRI-VET-AAABN",
        "county_TRI-VET-AAABT",
        "county_TRI-VET-AAABU",
        "county_TRI-VET-AAABW",
        "county_TRI-VET-AAABX",
        "county_TRI-VET-AAABY",
        "county_TRI-VET-AAACC",
        "county_TRI-VET-AAACF",
        "county_TRI-VET-AAACH",
        "county_TRI-VET-AAACW",
        "county_TRI-VET-AAACX",
        "county_TRI-VET-AAACZ",
        "county_TRI-VET-AAADB",
        "county_TRI-VET-AAADH",
        "county_TRI-VET-AAADJ",
        "county_TRI-VET-AAADK",
        "county_TRI-VET-AAADN",
        "county_TRI-VET-AAADR",
        "county_TRI-VET-AAADV",
        "county_TRI-VET-AAADY",
        "county_TRI-VET-AAADZ",
        "county_TRI-VET-AAAEA",
        "county_TRI-VET-AAAEP",
        "county_TRI-VET-AAAER",
        "county_TRI-VET-AAAET",
        "county_TRI-VET-AAAEW",
        "county_TRI-VET-AAAEZ",
        "county_TRI-VET-AAAFD",
        "county_TRI-VET-AAAFF",
        "county_TRI-VET-AAAFJ",
        "county_TRI-VET-AAAFK",
        "county_TRI-VET-AAAFL",
        "county_TRI-VET-AAAFR",
        "county_TRI-VET-AAAFS",
        "county_TRI-VET-AAAFW",
        "county_TRI-VET-AAAGA"
    };
    
    public static String[] HouseIncomeFiles = {
        "householdIncome_duabaaai",
        "householdIncome_DUABAAAL",
        "householdIncome_DUABAAAY",
        "householdIncome_DUABAABD",
        "householdIncome_DUABAABH",
        "householdIncome_DUABAABK",
        "householdIncome_DUABAABO",
        "householdIncome_DUABAABR",
        "householdIncome_duabaabs",
        "householdIncome_DUABAABV",
        "householdIncome_DUABAABW",
        "householdIncome_DUABAABY",
        "householdIncome_DUABAACA",
        "householdIncome_DUABAACB",
        "householdIncome_DUABAACD",
        "householdIncome_DUABAACG",
        "householdIncome_DUABAACH",
        "householdIncome_duabaaci",
        "householdIncome_DUABAACK",
        "householdIncome_DUABAACS",
        "householdIncome_DUABAACT",
        "householdIncome_DUABAACX",
        "householdIncome_DUACAAAA",
        "householdIncome_duacaaab",
        "householdIncome_DUACAAAC",
        "householdIncome_DUACAAAD",
        "householdIncome_DUACAAAE",
        "householdIncome_DUACAAAG",
        "householdIncome_DUACAAAK",
        "householdIncome_duacaaaq",
        "householdIncome_duacaaas",
        "householdIncome_DUACAAGD",
        "householdIncome_duacaahc",
        "householdIncome_gmacaacf",
        "householdIncome_soacaaai",
        "householdIncome_SOACAABI",
        "householdIncome_SOACAABL",
        "householdIncome_SOACAABM",
        "householdIncome_soacaabn",
        "householdIncome_SOACAABQ",
        "householdIncome_SOACAABR",
        "householdIncome_SOACAABX",
        "householdIncome_SOACAACA",
        "householdIncome_SOACAACL",
        "householdIncome_SOACAAEF",
        "householdIncome_SOACAAEL",
        "householdIncome_soacaafi",
        "householdIncome_SOACAAGH",
        "householdIncome_SOACAAGI",
        "householdIncome_SOADAAAY",
        "householdIncome_SOADAABA",
        "householdIncome_SOADAABB",
        "householdIncome_SOADAABF",
        "householdIncome_SOADAABI",
        "householdIncome_soadaabm",
        "householdIncome_soadaabr",
        "householdIncome_SOADAALV",
        "householdIncome_TRI-PPB-PETTY",
        "householdIncome_TRI-VET-AAAAA",
        "householdIncome_TRI-VET-AAAAB",
        "householdIncome_TRI-VET-AAAAC",
        "householdIncome_TRI-VET-AAAAD",
        "householdIncome_TRI-VET-AAAAE",
        "householdIncome_TRI-VET-AAAAF",
        "householdIncome_TRI-VET-AAAAG",
        "householdIncome_TRI-VET-AAAAJ",
        "householdIncome_TRI-VET-AAAAK",
        "householdIncome_TRI-VET-AAAAL",
        "householdIncome_TRI-VET-AAAAM",
        "householdIncome_TRI-VET-AAAAN",
        "householdIncome_TRI-VET-AAAAO",
        "householdIncome_TRI-VET-AAAAP",
        "householdIncome_TRI-VET-AAAAR",
        "householdIncome_TRI-VET-AAAAS",
        "householdIncome_TRI-VET-AAAAU",
        "householdIncome_TRI-VET-AAAAW",
        "householdIncome_TRI-VET-AAAAX",
        "householdIncome_TRI-VET-AAAAZ",
        "householdIncome_TRI-VET-AAABC",
        "householdIncome_TRI-VET-AAABD",
        "householdIncome_TRI-VET-AAABF",
        "householdIncome_TRI-VET-AAABH",
        "householdIncome_TRI-VET-AAABJ",
        "householdIncome_TRI-VET-AAABK",
        "householdIncome_TRI-VET-AAABL",
        "householdIncome_TRI-VET-AAABM",
        "householdIncome_TRI-VET-AAABN",
        "householdIncome_TRI-VET-AAABP",
        "householdIncome_TRI-VET-AAABT",
        "householdIncome_TRI-VET-AAABU",
        "householdIncome_TRI-VET-AAABW",
        "householdIncome_TRI-VET-AAABX",
        "householdIncome_TRI-VET-AAABY",
        "householdIncome_TRI-VET-AAACC",
        "householdIncome_TRI-VET-AAACE",
        "householdIncome_TRI-VET-AAACF",
        "householdIncome_TRI-VET-AAACH",
        "householdIncome_TRI-VET-AAACJ",
        "householdIncome_TRI-VET-AAACR",
        "householdIncome_TRI-VET-AAACW",
        "householdIncome_TRI-VET-AAACX",
        "householdIncome_TRI-VET-AAACZ",
        "householdIncome_TRI-VET-AAADB",
        "householdIncome_TRI-VET-AAADE",
        "householdIncome_TRI-VET-AAADH",
        "householdIncome_TRI-VET-AAADJ",
        "householdIncome_TRI-VET-AAADK",
        "householdIncome_TRI-VET-AAADL",
        "householdIncome_TRI-VET-AAADM",
        "householdIncome_TRI-VET-AAADN",
        "householdIncome_TRI-VET-AAADR",
        "householdIncome_TRI-VET-AAADS",
        "householdIncome_TRI-VET-AAADV",
        "householdIncome_TRI-VET-AAADW",
        "householdIncome_TRI-VET-AAADY",
        "householdIncome_TRI-VET-AAADZ",
        "householdIncome_TRI-VET-AAAEA",
        "householdIncome_TRI-VET-AAAEE",
        "householdIncome_TRI-VET-AAAEG",
        "householdIncome_TRI-VET-AAAEH",
        "householdIncome_TRI-VET-AAAEM",
        "householdIncome_TRI-VET-AAAEP",
        "householdIncome_TRI-VET-AAAER",
        "householdIncome_TRI-VET-AAAET",
        "householdIncome_TRI-VET-AAAEV",
        "householdIncome_TRI-VET-AAAEW",
        "householdIncome_TRI-VET-AAAEZ",
        "householdIncome_TRI-VET-AAAFB",
        "householdIncome_TRI-VET-AAAFC",
        "householdIncome_TRI-VET-AAAFD",
        "householdIncome_TRI-VET-AAAFF",
        "householdIncome_TRI-VET-AAAFG",
        "householdIncome_TRI-VET-AAAFJ",
        "householdIncome_TRI-VET-AAAFK",
        "householdIncome_TRI-VET-AAAFL",
        "householdIncome_TRI-VET-AAAFR",
        "householdIncome_TRI-VET-AAAFS",
        "householdIncome_TRI-VET-AAAFT",
        "householdIncome_TRI-VET-AAAFW",
        "householdIncome_TRI-VET-AAAGA"
    };
    
    public static String[] PropertyFiles = {
        "propertyType_duabaaai",
        "propertyType_DUABAAAL",
        "propertyType_DUABAAAY",
        "propertyType_DUABAABD",
        "propertyType_DUABAABH",
        "propertyType_DUABAABK",
        "propertyType_DUABAABO",
        "propertyType_DUABAABR",
        "propertyType_duabaabs",
        "propertyType_DUABAABV",
        "propertyType_DUABAABW",
        "propertyType_DUABAABY",
        "propertyType_DUABAACA",
        "propertyType_DUABAACB",
        "propertyType_DUABAACD",
        "propertyType_DUABAACG",
        "propertyType_DUABAACH",
        "propertyType_duabaaci",
        "propertyType_DUABAACS",
        "propertyType_DUABAACT",
        "propertyType_DUABAACX",
        "propertyType_DUACAAAA",
        "propertyType_DUACAAAD",
        "propertyType_DUACAAAE",
        "propertyType_DUACAAAG",
        "propertyType_DUACAAAK",
        "propertyType_duacaaaq",
        "propertyType_duacaaas",
        "propertyType_DUACAAGD",
        "propertyType_duacaahc",
        "propertyType_gmacaacf",
        "propertyType_SOACAABL",
        "propertyType_SOACAABM",
        "propertyType_SOACAABQ",
        "propertyType_SOACAABX",
        "propertyType_SOACAACA",
        "propertyType_SOACAACL",
        "propertyType_SOACAAEF",
        "propertyType_SOACAAEL",
        "propertyType_soacaafi",
        "propertyType_SOACAAGH",
        "propertyType_SOACAAGI",
        "propertyType_SOADAAAY",
        "propertyType_SOADAABA",
        "propertyType_SOADAABB",
        "propertyType_SOADAABF",
        "propertyType_SOADAABI",
        "propertyType_soadaabm",
        "propertyType_soadaabr",
        "propertyType_SOADAALV",
        "propertyType_TRI-VET-AAAAA",
        "propertyType_TRI-VET-AAAAB",
        "propertyType_TRI-VET-AAAAC",
        "propertyType_TRI-VET-AAAAD",
        "propertyType_TRI-VET-AAAAE",
        "propertyType_TRI-VET-AAAAF",
        "propertyType_TRI-VET-AAAAG",
        "propertyType_TRI-VET-AAAAJ",
        "propertyType_TRI-VET-AAAAK",
        "propertyType_TRI-VET-AAAAL",
        "propertyType_TRI-VET-AAAAM",
        "propertyType_TRI-VET-AAAAO",
        "propertyType_TRI-VET-AAAAP",
        "propertyType_TRI-VET-AAAAR",
        "propertyType_TRI-VET-AAAAU",
        "propertyType_TRI-VET-AAAAW",
        "propertyType_TRI-VET-AAAAX",
        "propertyType_TRI-VET-AAAAZ",
        "propertyType_TRI-VET-AAABC",
        "propertyType_TRI-VET-AAABD",
        "propertyType_TRI-VET-AAABF",
        "propertyType_TRI-VET-AAABH",
        "propertyType_TRI-VET-AAABJ",
        "propertyType_TRI-VET-AAABK",
        "propertyType_TRI-VET-AAABL",
        "propertyType_TRI-VET-AAABM",
        "propertyType_TRI-VET-AAABN",
        "propertyType_TRI-VET-AAABP",
        "propertyType_TRI-VET-AAABT",
        "propertyType_TRI-VET-AAABU",
        "propertyType_TRI-VET-AAABW",
        "propertyType_TRI-VET-AAABX",
        "propertyType_TRI-VET-AAABY",
        "propertyType_TRI-VET-AAACC",
        "propertyType_TRI-VET-AAACE",
        "propertyType_TRI-VET-AAACF",
        "propertyType_TRI-VET-AAACH",
        "propertyType_TRI-VET-AAACJ",
        "propertyType_TRI-VET-AAACR",
        "propertyType_TRI-VET-AAACW",
        "propertyType_TRI-VET-AAACX",
        "propertyType_TRI-VET-AAACZ",
        "propertyType_TRI-VET-AAADB",
        "propertyType_TRI-VET-AAADE",
        "propertyType_TRI-VET-AAADH",
        "propertyType_TRI-VET-AAADJ",
        "propertyType_TRI-VET-AAADK",
        "propertyType_TRI-VET-AAADL",
        "propertyType_TRI-VET-AAADM",
        "propertyType_TRI-VET-AAADN",
        "propertyType_TRI-VET-AAADR",
        "propertyType_TRI-VET-AAADS",
        "propertyType_TRI-VET-AAADV",
        "propertyType_TRI-VET-AAADW",
        "propertyType_TRI-VET-AAADY",
        "propertyType_TRI-VET-AAADZ",
        "propertyType_TRI-VET-AAAEA",
        "propertyType_TRI-VET-AAAEE",
        "propertyType_TRI-VET-AAAEG",
        "propertyType_TRI-VET-AAAEH",
        "propertyType_TRI-VET-AAAEM",
        "propertyType_TRI-VET-AAAEP",
        "propertyType_TRI-VET-AAAER",
        "propertyType_TRI-VET-AAAET",
        "propertyType_TRI-VET-AAAEV",
        "propertyType_TRI-VET-AAAEW",
        "propertyType_TRI-VET-AAAEZ",
        "propertyType_TRI-VET-AAAFB",
        "propertyType_TRI-VET-AAAFC",
        "propertyType_TRI-VET-AAAFD",
        "propertyType_TRI-VET-AAAFF",
        "propertyType_TRI-VET-AAAFG",
        "propertyType_TRI-VET-AAAFJ",
        "propertyType_TRI-VET-AAAFK",
        "propertyType_TRI-VET-AAAFL",
        "propertyType_TRI-VET-AAAFR",
        "propertyType_TRI-VET-AAAFT",
        "propertyType_TRI-VET-AAAFW",
        "propertyType_TRI-VET-AAAGA"
    };
    
    public static String[] YearFiles = {
        "yearBuilt_DUABAAAY",
        "yearBuilt_DUABAABD",
        "yearBuilt_DUABAABK",
        "yearBuilt_DUABAABO",
        "yearBuilt_DUABAABR",
        "yearBuilt_duabaabs",
        "yearBuilt_DUABAABW",
        "yearBuilt_DUABAABY",
        "yearBuilt_DUABAACA",
        "yearBuilt_DUABAACD",
        "yearBuilt_DUABAACG",
        "yearBuilt_DUABAACK",
        "yearBuilt_DUABAACS",
        "yearBuilt_DUABAACT",
        "yearBuilt_DUABAACX",
        "yearBuilt_DUACAAAA",
        "yearBuilt_duacaaab",
        "yearBuilt_DUACAAAC",
        "yearBuilt_DUACAAAD",
        "yearBuilt_DUACAAAE",
        "yearBuilt_DUACAAAG",
        "yearBuilt_duacaaas",
        "yearBuilt_DUACAAGD",
        "yearBuilt_gmacaacf",
        "yearBuilt_SOACAABI",
        "yearBuilt_SOACAABL",
        "yearBuilt_SOACAABM",
        "yearBuilt_soacaabn",
        "yearBuilt_SOACAABQ",
        "yearBuilt_SOACAABR",
        "yearBuilt_SOACAABX",
        "yearBuilt_SOACAACA",
        "yearBuilt_SOACAACL",
        "yearBuilt_SOACAAEF",
        "yearBuilt_soacaafi",
        "yearBuilt_SOACAAGH",
        "yearBuilt_SOACAAGI",
        "yearBuilt_SOADAABA",
        "yearBuilt_SOADAABB",
        "yearBuilt_SOADAABF",
        "yearBuilt_SOADAABI",
        "yearBuilt_soadaabm",
        "yearBuilt_soadaabr",
        "yearBuilt_SOADAALV",
        "yearBuilt_TRI-PPB-PETTY",
        "yearBuilt_TRI-VET-AAAAA",
        "yearBuilt_TRI-VET-AAAAB",
        "yearBuilt_TRI-VET-AAAAC",
        "yearBuilt_TRI-VET-AAAAD",
        "yearBuilt_TRI-VET-AAAAE",
        "yearBuilt_TRI-VET-AAAAG",
        "yearBuilt_TRI-VET-AAAAJ",
        "yearBuilt_TRI-VET-AAAAL",
        "yearBuilt_TRI-VET-AAAAM",
        "yearBuilt_TRI-VET-AAAAO",
        "yearBuilt_TRI-VET-AAAAR",
        "yearBuilt_TRI-VET-AAAAS",
        "yearBuilt_TRI-VET-AAAAU",
        "yearBuilt_TRI-VET-AAAAW",
        "yearBuilt_TRI-VET-AAAAX",
        "yearBuilt_TRI-VET-AAAAZ",
        "yearBuilt_TRI-VET-AAABC",
        "yearBuilt_TRI-VET-AAABD",
        "yearBuilt_TRI-VET-AAABF",
        "yearBuilt_TRI-VET-AAABH",
        "yearBuilt_TRI-VET-AAABK",
        "yearBuilt_TRI-VET-AAABL",
        "yearBuilt_TRI-VET-AAABM",
        "yearBuilt_TRI-VET-AAABN",
        "yearBuilt_TRI-VET-AAABT",
        "yearBuilt_TRI-VET-AAABW",
        "yearBuilt_TRI-VET-AAABX",
        "yearBuilt_TRI-VET-AAABY",
        "yearBuilt_TRI-VET-AAACC",
        "yearBuilt_TRI-VET-AAACE",
        "yearBuilt_TRI-VET-AAACF",
        "yearBuilt_TRI-VET-AAACH",
        "yearBuilt_TRI-VET-AAACJ",
        "yearBuilt_TRI-VET-AAACR",
        "yearBuilt_TRI-VET-AAACW",
        "yearBuilt_TRI-VET-AAACZ",
        "yearBuilt_TRI-VET-AAADB",
        "yearBuilt_TRI-VET-AAADE",
        "yearBuilt_TRI-VET-AAADH",
        "yearBuilt_TRI-VET-AAADJ",
        "yearBuilt_TRI-VET-AAADK",
        "yearBuilt_TRI-VET-AAADL",
        "yearBuilt_TRI-VET-AAADM",
        "yearBuilt_TRI-VET-AAADN",
        "yearBuilt_TRI-VET-AAADR",
        "yearBuilt_TRI-VET-AAADS",
        "yearBuilt_TRI-VET-AAADV",
        "yearBuilt_TRI-VET-AAADW",
        "yearBuilt_TRI-VET-AAADZ",
        "yearBuilt_TRI-VET-AAAEA",
        "yearBuilt_TRI-VET-AAAEE",
        "yearBuilt_TRI-VET-AAAEG",
        "yearBuilt_TRI-VET-AAAEH",
        "yearBuilt_TRI-VET-AAAEM",
        "yearBuilt_TRI-VET-AAAEP",
        "yearBuilt_TRI-VET-AAAER",
        "yearBuilt_TRI-VET-AAAET",
        "yearBuilt_TRI-VET-AAAEV",
        "yearBuilt_TRI-VET-AAAEW",
        "yearBuilt_TRI-VET-AAAEZ",
        "yearBuilt_TRI-VET-AAAFB",
        "yearBuilt_TRI-VET-AAAFC",
        "yearBuilt_TRI-VET-AAAFD",
        "yearBuilt_TRI-VET-AAAFF",
        "yearBuilt_TRI-VET-AAAFG",
        "yearBuilt_TRI-VET-AAAFJ",
        "yearBuilt_TRI-VET-AAAFK",
        "yearBuilt_TRI-VET-AAAFR",
        "yearBuilt_TRI-VET-AAAFS",
        "yearBuilt_TRI-VET-AAAFW"
    };

    public static void main(String[] args) {
        
        //CreateTrainTestData();
        
        //Instances train1 = ClassifierTools.loadData(storepath + county + "\\" + county + "_10train");
        //Instances test = ClassifierTools.loadData(storepath + county + "\\" + county + "_90test");
        
        //Instances train1 = ClassifierTools.loadData(storepath + householdIncome + "\\" + householdIncome + "_10train");
        //Instances test = ClassifierTools.loadData(storepath + householdIncome + "\\" + householdIncome + "_90test");
        
        //Instances train1 = ClassifierTools.loadData(storepath + propertyType + "\\" + propertyType + "_10train");
        //Instances test = ClassifierTools.loadData(storepath + propertyType + "\\" + propertyType + "_90test");
        
        Instances train1 = ClassifierTools.loadData(storepath + yearBuilt + "\\" + yearBuilt + "_10train");
        Instances test = ClassifierTools.loadData(storepath + yearBuilt + "\\" + yearBuilt + "_90test");
        
        //OutFile results = new OutFile(storepath + county + "\\" + county + "_1090rawdataResults.csv");
        //OutFile results = new OutFile(storepath + county + "\\" + county + "_1090acfResults.csv");
        //OutFile results = new OutFile(storepath + county + "\\" + county + "_1090psResults.csv");
        //OutFile results = new OutFile(storepath + county + "\\" + county + "_1090shapeletResults.csv");
        
        //OutFile results = new OutFile(storepath + householdIncome + "\\" + householdIncome + "_1090rawdataResults.csv");
        //OutFile results = new OutFile(storepath + householdIncome + "\\" + householdIncome + "_1090acfResults.csv");
        //OutFile results = new OutFile(storepath + householdIncome + "\\" + householdIncome + "_1090psResults.csv");
        //OutFile results = new OutFile(storepath + householdIncome + "\\" + householdIncome + "_1090shapeletResults.csv");
        
        //OutFile results = new OutFile(storepath + propertyType + "\\" + propertyType + "_1090rawdataResults.csv");
        //OutFile results = new OutFile(storepath + propertyType + "\\" + propertyType + "_1090acfResults.csv");
        //OutFile results = new OutFile(storepath + propertyType + "\\" + propertyType + "_1090psResults.csv");
        //OutFile results = new OutFile(storepath + propertyType + "\\" + propertyType + "_1090shapeletResults.csv");
        
        //OutFile results = new OutFile(storepath + yearBuilt + "\\" + yearBuilt + "_1090rawdataResults.csv");
        //OutFile results = new OutFile(storepath + yearBuilt + "\\" + yearBuilt + "_1090acfResults.csv");
        OutFile results = new OutFile(storepath + yearBuilt + "\\" + yearBuilt + "_1090psResults.csv");
        //OutFile results = new OutFile(storepath + yearBuilt + "\\" + yearBuilt + "_1090shapeletResults.csv");
        
        //Instances train = TransformExamples.acfTransform(train1);
        Instances train = TransformExamples.psTransform(train1);
        
        System.out.println("transform complete...");

        //Instances train = ClassifierTools.loadData(loadpath + "ElectricDevices_Train");
        //Instances test=ClassifierTools.loadData("C:\\Users\\RooTsMan v4.0 n.a.i\\Documents\\UEA\\Year 4\\MCOMP project\\experiments\\ElectricDevices\\ElectricDevices_Test");

        //LoadData(yearBuilt, YearFiles);
        //DataAnalysis();
        
        //Instances train = ClassifierTools.loadData(storepath + yearBuilt + "yearBuilt_train");
        //Instances test = ClassifierTools.loadData(storepath + yearBuilt + "yearBuilt_test");
        
        

        System.out.println("Training data length: " + train.numInstances());
        System.out.println("Testing data length: " + test.numInstances());
        //System.out.println("Whole data length: " + whole.numInstances());
        System.out.println();
        
        ArrayList<String> names = new ArrayList<String>();
        Classifier[] c = setSingleClassifiers(names);
        //OutFile results = new OutFile(storepath + yearBuilt + "yearBuilt_Results.csv");

        //double[] dist = null;
        //double[] classify_inst = new double[train.numInstances()];
        //double classify_inst;

        double[] correct = new double[c.length];

        for (int i = 0; i < c.length; i++) {

            double[] test_pred = new double[test.numInstances()];

            System.out.println("Classifier: " + names.get(i));
            try {
                c[i].buildClassifier(train);
                //} catch (Exception ex) {
                //    Logger.getLogger(ElectricityProfileClassification.class.getName()).log(Level.SEVERE, null, ex);
                //}
                System.out.println("Classifier " + names.get(i) + " built...");
                //try {
                //System.out.println(train.numInstances());
                //train.setClassIndex(train.numAttributes() - 1);
                //for (int j = 0; j < train.numInstances(); j++) {
                //if (j % 500 == 0) {
                //   System.out.println("j = " + j);
                //}
                //classify_inst[j] = c[i].classifyInstance(train.instance(j));
                //System.out.println(classify_inst);
                //dist = c[i].distributionForInstance(train.instance(j));
                //System.out.println(train.instance(j).classValue());
                //}
                
                //for(int blah = 0; blah < train.numInstances(); blah++){
                //    Instance inst = train.instance(blah);
                //    for(int other = 0; other < inst.numAttributes(); other++){
                //        if(Double.isInfinite(inst.value(other)) || Double.isNaN(inst.value(other))){
                //            System.out.println("Shouldn't have happened: " + blah + "; " + other );
                //        }
                //    }
                //}
                
                System.out.println("testing classifier...");
                for (int j = 0; j < test.numInstances(); j++) {
                    //System.out.println(j);
                    test_pred[j] = c[i].classifyInstance(test.instance(j));

                }
                System.out.println("checking accuracy...");
                for (int j = 0; j < test_pred.length; j++) {
                    if (test_pred[j] == test.instance(j).classValue()) {
                        correct[i]++;
                    }
                }

                //System.out.println("");
                //System.out.println("dist values:");

                //for (int k = 0; k < dist.length; k++) {
                //    System.out.println(dist[k]);
                //}

                //System.out.println("");
                //System.out.println("classification instance values:");
                //for (int k = 0; k < classify_inst.length; k++) {
                //    System.out.println(classify_inst[k]);
                //}
                System.out.println("");
            } catch (Exception ex) {
                Logger.getLogger(ElectricityProfileClassification.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        System.out.println("overall results;");
        System.out.println("");
        for (int i = 0; i < c.length; i++) {
            double acc = correct[i] / test.numInstances();
            results.writeLine(names.get(i) + "," + acc);
            System.out.println(names.get(i) + "," + acc);
        }

    }
    
    public static void CreateTrainTestData() {
        Instances countywhole = ClassifierTools.loadData(storepath + county + "\\" + county + "_whole");
        Instances householdIncomewhole = ClassifierTools.loadData(storepath + householdIncome + "\\" + householdIncome + "_whole");
        Instances propertyTypewhole = ClassifierTools.loadData(storepath + propertyType + "\\" + propertyType + "_whole");
        Instances yearBuiltwhole = ClassifierTools.loadData(storepath + yearBuilt + "\\" + yearBuilt + "_whole");
        
        System.out.println("data loaded...");
        
        OutFile countytrainfile=new OutFile(storepath+county+"\\"+county+"_40train.arff");  
        OutFile countytestfile=new OutFile(storepath+county+"\\"+county+"_60test.arff");
        
        OutFile hitrainfile=new OutFile(storepath+householdIncome+"\\"+householdIncome+"_40train.arff");  
        OutFile hitestfile=new OutFile(storepath+householdIncome+"\\"+householdIncome+"_60test.arff");
        
        OutFile pttrainfile=new OutFile(storepath+propertyType+"\\"+propertyType+"_40train.arff");  
        OutFile pttestfile=new OutFile(storepath+propertyType+"\\"+propertyType+"_60test.arff");
        
        OutFile ybtrainfile=new OutFile(storepath+yearBuilt+"\\"+yearBuilt+"_40train.arff");  
        OutFile ybtestfile=new OutFile(storepath+yearBuilt+"\\"+yearBuilt+"_60test.arff");
        
        System.out.println("files created...");
        
        countywhole.randomize(new Random());
        householdIncomewhole.randomize(new Random());
        propertyTypewhole.randomize(new Random());
        yearBuiltwhole.randomize(new Random());
        
        System.out.println("data randomized...");
        
        int countytrainSize = (int) Math.round(countywhole.numInstances() * 0.4);
        int countytestSize = countywhole.numInstances() - countytrainSize;
        
        int hitrainSize = (int) Math.round(householdIncomewhole.numInstances() * 0.4);
        int hitestSize = householdIncomewhole.numInstances() - hitrainSize;
        
        int pttrainSize = (int) Math.round(propertyTypewhole.numInstances() * 0.4);
        int pttestSize = propertyTypewhole.numInstances() - pttrainSize;
        
        int ybtrainSize = (int) Math.round(yearBuiltwhole.numInstances() * 0.4);
        int ybtestSize = yearBuiltwhole.numInstances() - ybtrainSize;
        
        System.out.println("train/test limits created...");
        
        Instances countytrain = new Instances(countywhole, 0, countytrainSize);
        Instances countytest = new Instances(countywhole, countytrainSize, countytestSize);
        
        Instances hitrain = new Instances(householdIncomewhole, 0, hitrainSize);
        Instances hitest = new Instances(householdIncomewhole, hitrainSize, hitestSize);
        
        Instances pttrain = new Instances(propertyTypewhole, 0, pttrainSize);
        Instances pttest = new Instances(propertyTypewhole, pttrainSize, pttestSize);
        
        Instances ybtrain = new Instances(yearBuiltwhole, 0, ybtrainSize);
        Instances ybtest = new Instances(yearBuiltwhole, ybtrainSize, ybtestSize);
        
        System.out.println("train/test instances made...");
        
        countytrainfile.writeString(countytrain.toString());
        countytestfile.writeString(countytest.toString());
        
        hitrainfile.writeString(hitrain.toString());
        hitestfile.writeString(hitest.toString());
        
        pttrainfile.writeString(pttrain.toString());
        pttestfile.writeString(pttest.toString());
        
        ybtrainfile.writeString(ybtrain.toString());
        ybtestfile.writeString(ybtest.toString()); 
        
        System.out.println("instances written to files...");
    }
    
     public static void DataAnalysis() {

        Instances train = ClassifierTools.loadData(storepath + yearBuilt + "yearBuilt_train");
        Instances test = ClassifierTools.loadData(storepath + yearBuilt + "yearBuilt_test");
        Instances whole = ClassifierTools.loadData(storepath + yearBuilt + "yearBuilt_whole");

        System.out.println("Training data length: " + train.numInstances());
        System.out.println("Testing data length: " + test.numInstances());
        System.out.println("Whole data length: " + whole.numInstances());
        System.out.println();
        int[] chk = new int[train.numClasses()];
        for(int i = 0; i < train.numInstances(); i++) {
            //System.out.println(whole.instance(i).classValue());
            for(int j = 0; j < train.numClasses(); j++) {
                if(train.instance(i).classValue() == j)
                    chk[j]++;
            }
        }
        
        for(int i = 0; i < chk.length; i++) {
            System.out.println("train: class "+i+" has "+ chk[i]+" values");
        }
        
        int[] chk1 = new int[test.numClasses()];
        for(int i = 0; i < test.numInstances(); i++) {
            //System.out.println(whole.instance(i).classValue());
            for(int j = 0; j < test.numClasses(); j++) {
                if(test.instance(i).classValue() == j)
                    chk1[j]++;
            }
        }
        
        for(int i = 0; i < chk1.length; i++) {
            System.out.println("test: class "+i+" has "+ chk1[i]+" values");
        }
        
        int[] chk2 = new int[whole.numClasses()];
        for(int i = 0; i < whole.numInstances(); i++) {
            //System.out.println(whole.instance(i).classValue());
            for(int j = 0; j < whole.numClasses(); j++) {
                if(whole.instance(i).classValue() == j)
                    chk2[j]++;
            }
        }
        
        for(int i = 0; i < chk2.length; i++) {
            System.out.println("whole: class "+i+" has "+ chk2[i]+" values");
        }
    }
    

    public static void LoadData(String Type, String[] Typefiles) {

        Instances data = ClassifierTools.loadData(loadpath + Type + Typefiles[0]);

        for (int i = 1; i < Typefiles.length; i++) {
            Instances file = ClassifierTools.loadData(loadpath + Type + Typefiles[i]);

            for (int j = 0; j < file.numInstances(); j++) {
                //System.out.println(file.numInstances());
                data.add(file.instance(j));
            }
        }
        
        TrainTestSplit(Type, data);

        //return data;
    }

    public static void TrainTestSplit(String type, Instances alldata) {
        
        OutFile combo=new OutFile(storepath+type+"yearBuilt_whole.arff");  
        OutFile trainfile=new OutFile(storepath+type+"yearBuilt_train.arff");  
        OutFile testfile=new OutFile(storepath+type+"yearBuilt_test.arff"); 
        
        alldata.randomize(new Random());

        //Instances alldata = ClassifierTools.loadData(storepath + type + "_whole");
        
        int trainSize = (int) Math.round(alldata.numInstances() * 0.5);
        int testSize = alldata.numInstances() - trainSize;
        
        Instances train = new Instances(alldata, 0, trainSize);
        Instances test = new Instances(alldata, trainSize, testSize);
        
        combo.writeString(alldata.toString());
        trainfile.writeString(train.toString());
        testfile.writeString(test.toString());
        

        //test.delete();
        //train.randomize(new Random());
        //int splitVal = train.numInstances() / 2;

        //for (int i = 0; i < splitVal; i++) {
        //    test.add(train.firstInstance());
        //    train.delete(0);
            
        //    System.out.println("Training data length: " + train.numInstances());
        //    System.out.println("Testing data length: " + test.numInstances());
        //}
        
        //return test;

    }

    public static Classifier[] setSingleClassifiers(ArrayList<String> names) {
        ArrayList<Classifier> sc2 = new ArrayList<>();
        sc2.add(new IBk(1));
        names.add("NN");
        Classifier c;
//		c=new DTW_kNN(1);
//		((DTW_kNN)c).setMaxR(0.01);

//		sc2.add(c);
//		names.add("NNDTW");
        sc2.add(new NaiveBayes());
        names.add("NB");
        sc2.add(new J48());
        names.add("C45");
        c = new SMO();
        PolyKernel kernel = new PolyKernel();
        kernel.setExponent(1);
        ((SMO) c).setKernel(kernel);
        sc2.add(c);
        names.add("SVML");
        c = new SMO();
        kernel = new PolyKernel();
        kernel.setExponent(2);
        ((SMO) c).setKernel(kernel);
        sc2.add(c);
        names.add("SVMQ");
        c = new SMO();
        c = new RandomForest();
        ((RandomForest) c).setNumTrees(30);
        sc2.add(c);
        names.add("RandF30");
        c = new RandomForest();
        ((RandomForest) c).setNumTrees(100);
        sc2.add(c);
        names.add("RandF100");
        c = new RandomForest();
        ((RandomForest) c).setNumTrees(500);
        sc2.add(c);
        names.add("RandF500");
        c = new RotationForest();
        sc2.add(c);
        names.add("RotF30");

        Classifier[] sc = new Classifier[sc2.size()];
        for (int i = 0; i < sc.length; i++) {
            sc[i] = sc2.get(i);
        }

        return sc;
    }
}
