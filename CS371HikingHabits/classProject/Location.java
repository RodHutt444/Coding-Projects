package classProject;

public class Location {
   
   private String location, zipcode, park;
   
   public Location(String l, String z, String p) {
      
      location = l;
      zipcode = z;
      park = p;
      
   } // end constructor
   
   public String getLocation() {
      return location;
   }
   
   public String getZipcode() {
      return zipcode;
   }
   
   public String getPark() {
      return park;
   }
   
   public void setLocation(String l) {
      location = l;
   }
   
   public void setZipcode(String z) {
      zipcode = z;
   }
   
   public void setPark(String p) {
      park = p;
   }
   
} // end class
