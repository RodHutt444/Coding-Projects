package classProject;

public class Profile {
   
   private String firstName, lastName, userName, password, email, aboutMe;
   
   public Profile(String f, String l, String u, String p, String e, String a) {
      
      firstName = f;
      lastName = l;
      userName = u;
      password = p;
      email = e;
      aboutMe = a;
      
   } // end constructor
   
   public String getFirstName() {
      return firstName;
   }
   
   public String getLastName() {
      return lastName;
   }
   
   public String getUserName() {
      return userName;
   }
   
   public String getPassword() {
      return password;
   }
   
   public String getEmail() {
      return email;
   }
   
   public String getAboutMe() {
      return aboutMe;
   }
   
   public void setFirstName(String f) {
      firstName = f;
   }
   
   public void setLastName(String l) {
      lastName = l;
   }
   
   public void setUserName(String u) {
      userName = u;
   }
   
   public void setPassword(String p) {
      password = p;
   }
   
   public void setEmail(String e) {
      email = e;
   }
   
   public void setAboutMe(String a) {
      aboutMe = a;
   }
   
} // end class   
