package classProject;
import java.sql.*;
import java.net.URL;
import java.util.ResourceBundle;
import javafx.application.Application;
import javafx.scene.control.TableColumn;
import javafx.fxml.FXMLLoader;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.event.ActionEvent;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.scene.Node;
import javafx.scene.input.MouseEvent;
import java.io.IOException;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TableRow;
import javafx.scene.control.TextField;
import javafx.scene.text.Text;
import javafx.scene.control.TextArea;
import javafx.scene.control.PasswordField;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;

public class sceneController implements Initializable { 
   
   private Stage stage;
   private Scene scene;
   private Parent root;
   private String userSearch;
   
   /* 
   * Tell TableView what will be inserted into resultTable and 
   * the tableColumns 
   */

   @FXML
   private TableView<routeSearchModel> resultTable;
   @FXML
   private TableColumn<routeSearchModel, String> routeDifficulty; 
   @FXML
   private TableColumn<routeSearchModel, String> routeLocation;
   @FXML
   private TableColumn<routeSearchModel, String> routeName;
   @FXML
   private TableColumn<routeSearchModel, String> idCol;
   
   /*
    * Object for search results.
    */
   @FXML
   private TextField searchText;
   
   /*
    * Objects for account creation/login/modification.
    */
   @FXML
   private TextField usernameText;
   @FXML
   private PasswordField passwordText;
   @FXML
   private TextField createFirstName;
   @FXML
   private TextField createLastName;
   @FXML
   private TextField createUserName;
   @FXML
   private PasswordField createPassword;
   @FXML
   private TextField createEmail;
   @FXML
   private TextArea userInfo;
   @FXML
   private TextField editFirstName;
   @FXML
   private TextField editLastName;
   @FXML
   private TextField editUserName;
   @FXML
   private PasswordField editPassword;
   @FXML
   private TextField editEmail;
   @FXML
   private TextArea editAboutMe;
   
  /*
   * Objects for location creation.
   */
   @FXML
   private TextField location;
   @FXML
   private TextField zipcode;
   private String park;
    
   /*
    * Objects for activity creation.
    */
   @FXML
   private TextField activityName;
   @FXML
   private TextField gpsCoord;
   @FXML
   private ChoiceBox activityTypeMenu;
   @FXML
   private TextField activityDifficulty;
    
   /*
    * Objects for activity page.
    */
   @FXML
   private Text aName;
   @FXML
   private Text aType;
   @FXML
   private Text rating;
   @FXML
   private Text locationInfo;
   @FXML
   private Text gpsString;
   @FXML
   private Text user;
   @FXML
   private Text date;
    
   /*
    * Objects for profileView page.
    */
   @FXML
   private Text fullName;
   @FXML
   private Text emailAddress;
   @FXML
   private Text userName;
   @FXML
   private TextArea aboutMeBio;
   
   /*
    * Objects for My Account page.
    */
   @FXML
   private Text myFirstName;
   @FXML
   private Text myLastName;
   @FXML
   private Text myUserName;
   @FXML
   private Text myEmail;
   @FXML
   private TextArea aboutMe;
    
    
   /*
    * Object for advanced search page.
    */
   @FXML
   private ChoiceBox difficultyType;
   @FXML
   private ChoiceBox trailType;
   @FXML
   private TextField locationString;
    
   /*
    * A list  that is used for the difficulty type dropdown menu on home scene.
    */
   ObservableList<String> aDifficultyTypeList = FXCollections.observableArrayList("1", "2", "3", "4", "5", "6", "7", "8", "9", "10");
    
   /*
    * A list  that is used for the trail type dropdown menu on home scene.
    */
   ObservableList<String> aTrailType = FXCollections.observableArrayList("Hiking Trail", "Biking Trail", "Climbing Wall", "Mountain Biking Trail");
   
   /*
    * A list that inputs data into the tableview.
    * Uses the routeSearchModel class as parameters.
    */
   ObservableList<routeSearchModel> routeSearchModelObservableList;
   
   /*
    * A list that is used for the activity type dropdown menu.
    */
   ObservableList<String> aTypeList = FXCollections.observableArrayList("Hiking Trail", "Biking Trail", "Climbing Wall", "Mountain Biking Trail");

   /*
   * Method: initialize
   * This method executes every time a new fxml file is loaded.
   */
   @Override
   public void initialize(URL url, ResourceBundle resource) {
   
      // This checks if the activity page was just loaded.
      if (aName != null) {
         
         // This attempts to load all the activty data into the activity page.
         try {
            ResultSet rs = retrieveData.getActivity(retrieveData.getActivityPageItem());
            try {
               if (rs.next()) {
                  aName.setText(rs.getString("activityName"));
                  aType.setText(rs.getString("aType"));
                  rating.setText(rs.getString("difficulty"));
                  locationInfo.setText(rs.getString("locationName") + ", " + rs.getString("zipCode"));
                  gpsString.setText(rs.getString("gpsCoord"));
                  user.setText(rs.getString("userName"));
                  date.setText(rs.getString("dateAdded"));
                  userSearch =rs.getString("userName");
                  retrieveData.setSearchUser(userSearch);
               } // end if
            } // end try
            catch (SQLException sqle) {
               System.out.println(sqle);
            } // end catch
         } // end try
         catch (IOException ioe) {
            System.out.println(ioe);
         } // end catch
         
      } // end if
   
      // This checks if the activity creation page was just loaded.
      if (activityTypeMenu != null) {
         activityTypeMenu.setValue("Select Type");
         activityTypeMenu.setItems(aTypeList); // This loads the options into the choice box.
      }
      
      // This checks if the home scene page was just loaded.
      if (difficultyType != null) {
         difficultyType.setValue("Select Difficulty");
         difficultyType.setItems(aDifficultyTypeList); // This loads the options into the choice box.
      }
      
      // This checks if the home scene page was just loaded.
      if (trailType != null) {
         trailType.setValue("Select Type");
         trailType.setItems(aTrailType); // This loads the options into the choice box.
      }

	   // Must check if the result table exists before info is added.
	   // Will return an error otherwise.
	   if (resultTable != null) {
		   routeDifficulty.setCellValueFactory(new PropertyValueFactory<>("Difficulty"));
		   routeLocation.setCellValueFactory(new PropertyValueFactory<>("Location"));
		   routeName.setCellValueFactory(new PropertyValueFactory<>("RouteName"));
         idCol.setCellValueFactory(new PropertyValueFactory<>("Id")); // This column remains invisible to the users.
         
         // This lambda function creates a clickable event for each row in the table view.
         resultTable.setRowFactory(tb -> {
            // This new table row is loaded with the data that was clicked on.
            TableRow<routeSearchModel> row = new TableRow<routeSearchModel>() {
               @Override
               protected void updateItem(routeSearchModel item, boolean empty) {
                  super.updateItem(item, empty);
                  setItem(empty ? null : item);
               } // end updateitem
            };
            // Listens for a mouse click.
            row.setOnMouseClicked(e -> {
               // Checks if row is empty.
               if (!row.isEmpty()) {
                  routeSearchModel item = row.getItem();
                  try {
                     retrieveData.setActivityPageItem(item); // Sends item to retrieveData.java.
                     activityPage(e); // Calls activityPage() method.
                  }
                  catch (IOException ioe) {
                     System.out.println(ioe);
                  }
               } // end if
            });
            return row;
         });
         
		   // Uses the retrieveData class to get a list from the database based on the search string.
		   routeSearchModelObservableList = retrieveData.getList();
         if (!retrieveData.getAdvancedSearch().equals(""))
            routeSearchModelObservableList = retrieveData.getListAdvanced();
		   // Inputs the list into the tableview.
		   resultTable.setItems(routeSearchModelObservableList);
	   } // end if
      
      // Checks if the ProfileView page was just loaded.
      if (fullName != null) {
      
         //Gets the username that was displayed
         userSearch = retrieveData.getSearchUser();
         
         // Tries to load data of user to profileView page
         try {
         
            ResultSet res = retrieveData.getProfile(userSearch);
            try {
            
               if (res.next()) {
               
                  userName.setText(res.getString("userName"));
                  fullName.setText(res.getString("firstName") + " " + res.getString("lastName"));
                  emailAddress.setText(res.getString("email"));
                  aboutMeBio.setText(res.getString("aboutMe"));
                  
               }//end if
            } // end try
            catch (SQLException sqle) {
               System.out.println(sqle);
            } // end catch
         } // end try
         catch (IOException ioe) {
            System.out.println(ioe);
         } // end catch
         
      } // end if
      
      if (editFirstName != null) {
         try {
            ResultSet rs = retrieveData.getProfile(retrieveData.getAccountUsername());
            try {
               if (rs.next()) {
                  editFirstName.setText(rs.getString("firstName"));
                  editLastName.setText(rs.getString("lastName"));
                  editUserName.setText(rs.getString("userName"));
                  editEmail.setText(rs.getString("email"));
                  editAboutMe.setText(rs.getString("aboutMe"));
               }
            }
            catch (SQLException sqle) {
               System.out.println(sqle);
            }
         }
         catch (IOException ioe) {
            System.out.println(ioe);
         }
      } // end if
      
      if (myFirstName != null) {
         try {
            ResultSet rs = retrieveData.getProfile(retrieveData.getAccountUsername());
            try {
               if (rs.next()) {
                  myFirstName.setText(rs.getString("firstName"));
                  myLastName.setText(rs.getString("lastName"));
                  myUserName.setText(rs.getString("userName"));
                  myEmail.setText(rs.getString("email"));
                  aboutMe.setText(rs.getString("aboutMe"));
               }
            }
            catch (SQLException sqle) {
               System.out.println(sqle);
            }
         }
         catch (IOException ioe) {
            System.out.println(ioe);
         }
      } // end if
                
   } // end initialize
   
   /*  
   * Method : homeScene
   * This method loads the home scene.
   */
   public void homeScene(ActionEvent event) throws IOException { 
      retrieveData.setLoggedIn(false);
      Parent root = FXMLLoader.load(getClass().getResource("hikingHabitsUI.fxml"));
      stage = (Stage)((Node)event.getSource()).getScene().getWindow();
      scene = new Scene(root);
      stage.setScene(scene);
      stage.show();
   } // end homeScene


    /*
   * Method: loggedInScene
   * This method loads the logged in scene after login.
   */
   public void loggedInScene(ActionEvent event) throws IOException { 
      
      // Creates an activity if this method was triggered by the activityCreation scene.
      if (activityTypeMenu != null) {
         Activity a = new Activity(activityName.getText(), gpsCoord.getText(), activityTypeMenu.getValue().toString(), activityDifficulty.getText());
         retrieveData.createActivity(a);
      }
      
      // Loads the username and password into retrieveData.java if possible.
      if (usernameText != null) {
         retrieveData.setAccountUsername(usernameText.getText());
         retrieveData.setAccountPassword(passwordText.getText());
      }
	//Modifies profile values if editProfile loads this scene
      if (editUserName != null || editFirstName != null || editLastName != null || editEmail != null || editPassword != null) {
         Profile q = new Profile(editFirstName.getText(), editLastName.getText(), editUserName.getText(), editPassword.getText(), editEmail.getText(), editAboutMe.getText());
         retrieveData.mutateProfile(q);
      }

         
      retrieveData.setLoggedIn(false);
      Parent root = FXMLLoader.load(getClass().getResource("hikingHabitsUI.fxml"));
      // Only loads logged in scene if user credentials exist.
      if (retrieveData.exists())
         root = FXMLLoader.load(getClass().getResource("userLogin.fxml"));
      stage = (Stage)((Node)event.getSource()).getScene().getWindow();
      scene = new Scene(root);
      stage.setScene(scene);
      stage.show();
      
   } // end loggedInScene

   /*
   * Method: resultScene
   * This method loads the result scene after the search bar is used.
   */
   public void resultScene(ActionEvent event) throws IOException {
      retrieveData.setAdvancedSearch("");
	   // Loads the text from the search into retrieveData class.
       if (searchText != null)
	       retrieveData.setSearch(searchText.getText());
       root = FXMLLoader.load(getClass().getResource("resultScene.fxml"));
       stage = (Stage)((Node)event.getSource()).getScene().getWindow();
       scene = new Scene(root);
       stage.setScene(scene);
       stage.show();
   } // end resultScene
   
   /*
    * Method: resultSceneNotLoggedIn
    * This method loads a result scene when the user hasn't logged in yet.
    */
   public void resultSceneNotLoggedIn(ActionEvent event) throws IOException {
      retrieveData.setLoggedIn(false);
      retrieveData.setAdvancedSearch("");
	   // Loads the text from the search into retrieveData class.
      if (searchText != null)
	      retrieveData.setSearch(searchText.getText());
	   root = FXMLLoader.load(getClass().getResource("resultSceneNotLoggedIn.fxml"));
	   stage = (Stage)((Node)event.getSource()).getScene().getWindow();
	   scene = new Scene(root);
	   stage.setScene(scene);
	   stage.show();
   } // end resultSceneNotLoggedIn
   
   public void resultSceneAdvanced(ActionEvent event) throws IOException {
   
      if (locationString != null) {
      
         String advancedDiff = difficultyType.getValue().toString();
         if (advancedDiff.equals("Select Difficulty"))
            advancedDiff = "";
            
         String advancedType = trailType.getValue().toString();
         if (advancedType.equals("Select Type"))
            advancedType = "";
            
         String advancedLocation = locationString.getText();
         String advancedSearch = "";
         
         if (advancedDiff.equals("")) {
             advancedSearch = "select *\n" +
	                  	   "from Activity A natural join Location L\n" +
	                  	   "where (A.aType like '%" + advancedType + "%') and (L.locationName like '%" + advancedLocation + "%')\n" +
	                  	   "order by A.activityName asc;";
         } // end if
         
         else {
            advancedSearch = "select *\n" +
	                  	"from Activity A natural join Location L\n" +
	                  	"where (A.aType like '%" + advancedType + "%') and (L.locationName like '%" + advancedLocation + "%') and (A.difficulty = " + advancedDiff + ")\n" +
	                  	"order by A.activityName asc;";
         }
         
         retrieveData.setAdvancedSearch(advancedSearch);
         
      } // end if
   
      if (retrieveData.isLoggedIn()) 
         root = FXMLLoader.load(getClass().getResource("resultScene.fxml"));
      else
         root = FXMLLoader.load(getClass().getResource("resultSceneNotLoggedIn.fxml"));
      stage = (Stage)((Node)event.getSource()).getScene().getWindow();
	   scene = new Scene(root);
	   stage.setScene(scene);
	   stage.show();
      
   }
   
   /*
    * Method: profileLogin
    * This method loads a page that allows a user to log in to their profile.
    */
   public void profileLogin (ActionEvent event) throws IOException {
      // Creates a profile if the profile creation page is loading this one.
      if (createUserName != null) {
         Profile p = new Profile(createFirstName.getText(), createLastName.getText(), createUserName.getText(), createPassword.getText(), createEmail.getText(), userInfo.getText());
         retrieveData.createProfile(p);
      }
      root = FXMLLoader.load(getClass().getResource("profileLogin.fxml"));
      stage = (Stage)((Node)event.getSource()).getScene().getWindow();
	   scene = new Scene(root);
	   stage.setScene(scene);
	   stage.show();
      
   } // end profileLogin
   
   /*
    * Method: uploadActivity
    * This method loads a scene after the user enters location information
    */
   public void uploadActivity (ActionEvent event) throws IOException {
   
      // Creates a location based on the info given in the previous scene.
      Location l = new Location(location.getText(), zipcode.getText(), park);
      retrieveData.createLocation(l);
      
      root = FXMLLoader.load(getClass().getResource("uploadActivity.fxml"));
      stage = (Stage)((Node)event.getSource()).getScene().getWindow();
	   scene = new Scene(root);
	   stage.setScene(scene);
	   stage.show();
      
   } // end uploadActivity
   
   /*
    * Method: createProfile
    * This method loads a scene to create new Account/Profile
    */
   public void createProfile (ActionEvent event) throws IOException {
   
      root = FXMLLoader.load(getClass().getResource("createProfile.fxml"));
      stage = (Stage)((Node)event.getSource()).getScene().getWindow();
	   scene = new Scene(root);
	   stage.setScene(scene);
	   stage.show();
      
   } // end createProfile
   
   /*
    * Method: uploadActivitLocation
    * This method loads a scene to create a new Activity (specifically its location)
    */
   public void uploadActivityLocation (ActionEvent event) throws IOException {
   
      root = FXMLLoader.load(getClass().getResource("uploadActivityLocation.fxml"));
      stage = (Stage)((Node)event.getSource()).getScene().getWindow();
	   scene = new Scene(root);
	   stage.setScene(scene);
	   stage.show();
      
   } // end uploadActivitLocation
   
   /*
    * Method: isPark
    * This method sets park string to True.
    */
   public void isPark() {
      park = "True";
   } // end isPark
   
   /*
    * Method: isNotPark
    * This method sets park string to False.
    */
   public void isNotPark() {
      park = "False";
   } // end isNotPark
   
   /*
    * Method activityPage
    * This method loads the activity page with the necessary data.
    */
   public void activityPage(MouseEvent event) throws IOException {
      
      // Loads different scenes based on if user is logged in.
      if (retrieveData.isLoggedIn())
         root = FXMLLoader.load(getClass().getResource("activityPage.fxml"));
      else {
         retrieveData.setLoggedIn(false);
         root = FXMLLoader.load(getClass().getResource("activityPageNotLoggedIn.fxml"));
      }
      stage = (Stage) ((Node) event.getSource()).getScene().getWindow();
      scene = new Scene(root);
      stage.setScene(scene);
      stage.show();
      
   } // end activityPage
   
   /*
    * Method: profileViewNotLoggedIn
    * This method loads a scene to view a profiles data like name, email, etc when not logged in.
    */
   public void profileViewNotLoggedIn (ActionEvent event) throws IOException {
   
      retrieveData.setLoggedIn(false);
      root = FXMLLoader.load(getClass().getResource("profileViewNotLoggedIn.fxml"));
      stage = (Stage)((Node)event.getSource()).getScene().getWindow();
	   scene = new Scene(root);
	   stage.setScene(scene);
	   stage.show();
      
   } // end uploadActivitLocation
   
   /*
    * Method: profileView
    * This method loads a scene to view a profiles data like name, email, etc when logged in.
    */
   public void profileView (ActionEvent event) throws IOException {
   
      root = FXMLLoader.load(getClass().getResource("profileView.fxml"));
      stage = (Stage)((Node)event.getSource()).getScene().getWindow();
	   scene = new Scene(root);
	   stage.setScene(scene);
	   stage.show();
      
   } // end uploadActivitLocation
	
   public void editProfile (ActionEvent event) throws IOException {
   
	   root = FXMLLoader.load(getClass().getResource("editProfile.fxml"));
	   stage = (Stage)((Node)event.getSource()).getScene().getWindow();
	   scene = new Scene(root);
	   stage.setScene(scene);
	   stage.show();
      
   } // end editProfile
   
   public void myAccount(ActionEvent event) throws IOException {
      
      root = FXMLLoader.load(getClass().getResource("myProfile.fxml"));
      stage = (Stage) ((Node) event.getSource()).getScene().getWindow();
      scene = new Scene(root);
      stage.setScene(scene);
      stage.show();
      
   } // end myAccount
   
} // end class 
