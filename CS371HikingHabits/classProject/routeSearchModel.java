package classProject;

import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.SimpleIntegerProperty;

public class routeSearchModel { 

   SimpleStringProperty id, routeName, location;
   SimpleIntegerProperty difficulty;
   
   public routeSearchModel(String id, String routeName, String difficulty, String location) { 
      this.routeName = new SimpleStringProperty(routeName);
      this.difficulty = new SimpleIntegerProperty(Integer.parseInt(difficulty));
      this.location = new SimpleStringProperty(location);
      this.id = new SimpleStringProperty(id);
   }
   
   public String getRouteName() { 
      return routeName.get();
   }
   
   public int getDifficulty() { 
      return difficulty.get();
   }
   
   public String getLocation() { 
      return location.get();
   }
   
   public String getId() {
      return id.get();
   }
   
   public void setRouteName(String routeName) { 
      this.routeName = new SimpleStringProperty(routeName);
   }
   
   public void setDifficulty(String difficulty) { 
      this.difficulty = new SimpleIntegerProperty(Integer.parseInt(difficulty));
   }
   
   public void setLocation (String location) { 
      this.location = new SimpleStringProperty(location);
   }
   
   public void setId(String id) {
      this.id = new SimpleStringProperty(id);
   }
   
} // end class 