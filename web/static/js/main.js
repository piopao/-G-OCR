$(document).ready(function() { 
    // $( "#submit" ).click(function() {
    //     var myData = "This is my data string."
    //     // $.post("/receivedata", {"myData": myData})
    //     console.log("STSRS");
    //     var myFile = $('#input_file').prop('files');
    //     console.log(myFile);
    // });

    console.log("RAME")

    function readURL(input) {

      if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function(e) {
          $('#pic').attr('src', e.target.result);
        }

        reader.readAsDataURL(input.files[0]);
      }
    }
  
    $("#file_input").change(function (){
      console.log("asdasd")
      $picture = $('#pic');
      $picture.removeClass("hidden");
      readURL(this);
      $download = $("#output_file");
      $download.addClass("hidden");
    });
  
    function add_download(href) {
      $download = $("#output_file");
      $download.removeClass("hidden");
      $download.attr("href", href);
    }

    $('#submit').click(function() {
        console.log("VQENI")
        event.preventDefault();
        var form_data = new FormData($('#uploadform')[0]);
        console.log(form_data);
        $.ajax({
            type: 'POST',
            url: '/uploadajax',
            data: form_data,
            contentType: false,
            processData: false,
            dataType: 'json'
        }).done(function(data, textStatus, jqXHR){
            console.log("chemikargimovtyan")
            console.log(data);
            console.log(textStatus);
            console.log(jqXHR);
            console.log('Success!');
            alert(data['url'])
            add_download(data['url']);
            // console.log(data);
            // $("#resultFilesize").text(data['size']);
        }).fail(function(data){
            alert('error!');
        });
    });
   
});








