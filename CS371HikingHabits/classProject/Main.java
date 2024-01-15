package classProject;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.scene.Parent;

public class Main extends Application { 

    /*
     * Method: start
     * This method is the main driver of the program. 
     * It initiates the homeScene on launch.
     */
	@Override
    public void start(Stage stage) { 
		
        try {
        	
        	// Loads the home page.
            Parent root = FXMLLoader.load(getClass().getResource("hikingHabitsUI.fxml"));
            Scene scene = new Scene(root);
            stage.setScene(scene);
            stage.show();
            
        } catch (Exception e) { 
        	
            e.printStackTrace();
            
        } // end catch
   } // end start

   public static void main(String[] args) {
	   
       launch(args);
       
   } 

} 