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
 * FileLogger.java
 * Copyright (C) 2008-2012 University of Waikato, Hamilton, New Zealand
 */

package weka.core.logging;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Date;
import java.util.regex.Matcher;

import weka.core.RevisionUtils;
//import weka.core.WekaPackageManager;

/**
 * A simple file logger, that just logs to a single file. Deletes the file
 * when an object gets instantiated.
 * 
 * @author  fracpete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 8034 $
 */
public class FileLogger
  extends ConsoleLogger {

  /** the log file. */
  protected File m_LogFile;
  
  /** the line feed. */
  protected String m_LineFeed;
  
  /**
   * Initializes the logger.
   */
  protected void initialize() {
    super.initialize();

    // log file
    m_LogFile = getLogFile();
    // try to remove file
    try {
      if ((m_LogFile != null) && m_LogFile.exists())
	m_LogFile.delete();
    }
    catch (Exception e) {
      e.printStackTrace();
    }
    
    // the line feed
    m_LineFeed = System.getProperty("line.separator");
  }
  
  /**
   * Returns the log file to use.
   * 
   * @return		the log file
   */
  protected File getLogFile() {
    String	filename;
    File	result;
    
    filename = m_Properties.getProperty("LogFile", "%w" + File.separator + "weka.log");
    filename = filename.replaceAll("%t", Matcher.quoteReplacement(System.getProperty("java.io.tmpdir")));
    filename = filename.replaceAll("%h", Matcher.quoteReplacement(System.getProperty("user.home")));
    filename = filename.replaceAll("%c", Matcher.quoteReplacement(System.getProperty("user.dir")));
//    filename = filename.replaceAll("%w", Matcher.quoteReplacement(WekaPackageManager.WEKA_HOME.toString()));
    if (System.getProperty("%") != null && System.getProperty("%").length() > 0) {
      filename = filename.replaceAll("%%", Matcher.quoteReplacement(System.getProperty("%")));
    }
    
    result = new File(filename);
    
    return result;
  }
  
  /**
   * Appends the given string to the log file (without new line!).
   * 
   * @param s		the string to append
   */
  protected void append(String s) {
    BufferedWriter	writer;
   
    if (m_LogFile == null)
      return;
    
    // append output to file
    try {
      writer = new BufferedWriter(new FileWriter(m_LogFile, true));
      writer.write(s);
      writer.flush();
      writer.close();
    }
    catch (Exception e) {
      // ignored
    }
  }

  /**
   * Performs the actual logging. 
   * 
   * @param level	the level of the message
   * @param msg		the message to log
   * @param cls		the classname originating the log event
   * @param method	the method originating the log event
   * @param lineno	the line number originating the log event
   */
  protected void doLog(Level level, String msg, String cls, String method, int lineno) {
    // output to console
    super.doLog(level, msg, cls, method, lineno);
    
    // append output to file
    append(
	m_DateFormat.format(new Date()) + " " + cls + " " + method + m_LineFeed
	+ level + ": " + msg + m_LineFeed);
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
