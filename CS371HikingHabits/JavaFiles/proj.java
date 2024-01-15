import java.sql.*;
import java.io.*;
import java.util.*;

public class proj {
	
	public static void RetrieveAllData(String address, String userID, String password) {
		
		Scanner scan = new Scanner(System.in);
		
		try {
			
			Connection conn = DriverManager.getConnection(address, userID, password);
			Statement stmt = conn.createStatement();
			
			try {
				
				ResultSet res = stmt.executeQuery(
						"select A.activityName, A.aType, A.difficulty, L.locationName, L.zipCode, L.Park, A.gpsCoord\n" +
                  "from Activity A natural join Location L\n" +
                  "order by A.activityName asc;");
				
				ResultSetMetaData rsmd = res.getMetaData();
				int numCol = rsmd.getColumnCount();
				String colNames[] = new String[numCol];
				int colSizes[] = new int[numCol];
				
				System.out.println();
				
				for (int i = 1; i <= numCol; i++) {
					colNames[i - 1] = rsmd.getColumnName(i);
					colSizes[i - 1] = (rsmd.getColumnDisplaySize(i) > colNames[i - 1].length() ? rsmd.getColumnDisplaySize(i) + 1 : colNames[i - 1].length() + 1);
					if (colSizes[i - 1] > 50)
						colSizes[i - 1] = 50;
					String format = "%-" + colSizes[i - 1] + "s";
					System.out.printf(format, colNames[i - 1]);
				} // end for
				
				System.out.println();
					
				for (int i = 0; i < numCol; i++)
					for (int j = 0; j < colSizes[i]; j++)
						System.out.print("-");
				
				System.out.println();
				
				while (res.next()) {
					
					for (int i = 0; i < numCol; i++) {
						String format = "%-" + colSizes[i] + "s";
						System.out.printf(format, res.getString(i + 1));
					} // end for
					
					System.out.println();
					
				} // end while
				
			} // end try
			
			catch (SQLException sql) {
				System.out.println("Could not retrieve data: " + sql);
			} // end catch
			
			stmt.close();
			conn.close();
			
		} // end try
		
		catch (Exception sqle) {
			
			System.out.println("Error: " + sqle);
			
		} // end catch
		
	} // end RetrieveAllData
	
	public static void Search(String address, String userID, String password) {
		
		Scanner scan = new Scanner(System.in);
		System.out.print("Please enter the search string: ");
		String searchString = scan.nextLine();
      if (searchString.charAt(searchString.length() - 1) == '\n')
         searchString = searchString.substring(0, searchString.length() - 1);
		
		try {
			
			Connection conn = DriverManager.getConnection(address, userID, password);
			Statement stmt = conn.createStatement();
			
			try {
				
				ResultSet res = stmt.executeQuery(
						"select A.activityName, A.aType, A.difficulty, L.locationName, L.zipCode, L.Park, A.gpsCoord\n" +
                  "from Activity A natural join Location L\n" +
                  "where (A.activityName like '%" + searchString + "%') or (A.aType like '%" + searchString + "%') or (L.locationName like '%" + searchString + "%')" +
                  "order by A.activityName asc;");
				
				ResultSetMetaData rsmd = res.getMetaData();
				int numCol = rsmd.getColumnCount();
				String colNames[] = new String[numCol];
				int colSizes[] = new int[numCol];
				
				System.out.println();
				
				for (int i = 1; i <= numCol; i++) {
					colNames[i - 1] = rsmd.getColumnName(i);
					colSizes[i - 1] = (rsmd.getColumnDisplaySize(i) > colNames[i - 1].length() ? rsmd.getColumnDisplaySize(i) + 1 : colNames[i - 1].length() + 1);
					if (colSizes[i - 1] > 50)
						colSizes[i - 1] = 50;
					String format = "%-" + colSizes[i - 1] + "s";
					System.out.printf(format, colNames[i - 1]);
				} // end for
				
				System.out.println();
					
				for (int i = 0; i < numCol; i++)
					for (int j = 0; j < colSizes[i]; j++)
						System.out.print("-");
				
				System.out.println();
				
				while (res.next()) {
					
					for (int i = 0; i < numCol; i++) {
						String format = "%-" + colSizes[i] + "s";
						System.out.printf(format, res.getString(i + 1));
					} // end for
					
					System.out.println();
					
				} // end while
				
			} // end try
			
			catch (SQLException sql) {
				System.out.println("Could not retrieve data: " + sql);
			} // end catch
			
			stmt.close();
			conn.close();
			
		} // end try
		
		catch (Exception sqle) {
			
			System.out.println("Error: " + sqle);
			
		} // end catch
		
	} // end RetrieveAllData
	
	public static void Insert(String address, String userID, String password) {
		
		Scanner scan = new Scanner(System.in);
		System.out.print("Into which table would you like to insert data (Use any necessary SQL syntax)? ");
		String tableName = scan.nextLine();
		
		try {
			
			
			Connection conn = DriverManager.getConnection(address, userID, password);
			Statement stmt = conn.createStatement();
			
			try {
				
				ResultSet res = stmt.executeQuery(
						"select *" +
				        "from " + tableName);
				  
				ResultSetMetaData rsmd = res.getMetaData();
				int numCol = rsmd.getColumnCount();
				String data = "";
				
				System.out.print("Columns in " + tableName + ": ");
				
				for (int i = 1; i <= numCol; i++)
					System.out.print(rsmd.getColumnName(i) + " ");
				
				System.out.println("\n--------------------------------------------------------------------------------");
				System.out.print("Please input SQL Insert statement: ");
				data = scan.nextLine();
				
				stmt.executeUpdate(data);
				System.out.println("Data inserted successfully.");
				
			} // end try
			
			catch (SQLException sql) {
				
				System.out.println("Could not insert data: " + sql);
				
			} // end catch
			
			stmt.close();
			conn.close();
			
		} // end try
		
		catch (Exception sqle) {
			
			System.out.println("Error: " + sqle);
			
		} // end catch
		
	} // end Insert
	
	public static void Delete(String address, String userID, String password) {
		
		Scanner scan = new Scanner(System.in);
		System.out.print("From which table would you like to delete from (Use any necessary SQL syntax)? ");
		String tableName = scan.nextLine();
		
		try {
			
			Connection conn = DriverManager.getConnection(address, userID, password);
			Statement stmt = conn.createStatement();
			
			try {
				
				ResultSet res = stmt.executeQuery(
						"select *" +
				        "from " + tableName);
				
				ResultSetMetaData rsmd = res.getMetaData();
				int colNum = rsmd.getColumnCount();
				String data = "";
				
				System.out.print("Columns in " + tableName + ": ");
				
				for (int i = 1; i <= colNum; i++)
					System.out.print(rsmd.getColumnName(i) + " ");
				
				System.out.println("\n--------------------------------------------------------------------------------");
				System.out.print("Please input SQL Delete statement: ");
				data = scan.nextLine();
				
				stmt.executeUpdate(data);
				System.out.println("Data successfully deleted.");
				
			} // end try
			
			catch (SQLException sql) {
				
				System.out.println("Data could not be deleted: " + sql);
				
			} // end catch
			
			stmt.close();
			conn.close();
			
		} // end try
		
		catch (Exception sqle) {
			
			System.out.println("Error: " + sqle);
			
		} // end catch
		
	} // end Delete

	public static void main(String[] args) {
		
		Scanner scan = new Scanner(System.in);
		
		System.out.printf("\nWhat would you like to do? (RETRIEVE ALL ACTIVITIES, SEARCH, INSERT, DELETE, EXIT): ");
		String operation = scan.nextLine();
		
		String userID = "root";
		String password = "CS371HikingHabits";
		String address = "jdbc:mysql://127.0.0.1:3306/hikinghabits_schema";
		
		while (operation.compareToIgnoreCase("EXIT") != 0) {
			
			if (operation.compareToIgnoreCase("RETRIEVE ALL ACTIVITIES") == 0)
				RetrieveAllData(address, userID, password);
			else if (operation.compareToIgnoreCase("SEARCH") == 0)
				Search(address, userID, password);
			else if (operation.compareToIgnoreCase("INSERT") == 0)
				Insert(address, userID, password);
			else if (operation.compareToIgnoreCase("DELETE") == 0)
				Delete(address, userID, password);
			else
				System.out.println("Command not recognized.");
			
			System.out.printf("\nWhat would you like to do? (RETRIEVE ALL ACTIVITIES, SEARCH, INSERT, DELETE, EXIT): ");
			operation = scan.nextLine();
			
		} // end while

	} // end main

} // end class
