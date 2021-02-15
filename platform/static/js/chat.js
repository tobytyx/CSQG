
var img_path = '';
var record_id = '';
var start_chat = false;

(
  function(){
    var chat = {
      messageToSend: '',
      init: function() {
        this.cacheDOM();
        this.bindEvents();
        this.render();
      },

      cacheDOM: function() {
        this.$chatHistory = $('.chat-history');
        this.$button = $('#message-button');
        this.$start = $('#start-button');
        this.$textarea = $('#message-to-send');
        this.$chatHistoryList =  this.$chatHistory.find('ul');
      },

      bindEvents: function() {
        this.$button.on('click', this.addMessage.bind(this));
        this.$start.on('click', this.startMessage.bind(this));
        this.$textarea.on('keyup', this.addMessageEnter.bind(this));
      },

      render: function() {
        this.scrollToBottom();
        if (this.messageToSend.trim() !== '') {
          var template = Handlebars.compile( $("#message-template").html());
          var context = { 
            messageOutput: this.messageToSend,
            time: this.getCurrentTime()
          };

          this.$chatHistoryList.append(template(context));
          this.scrollToBottom();
          this.$textarea.val('');
          // responses
          
          var query = "";
          $.ajax({
            type: 'POST',
            async: false,
            url: "/request/query/",
            data: {"user_id": user_id, "img_name": img_path, "record_id": record_id},
            success: function(rdata){
              query = rdata["query"];
              record_id = rdata["record_id"];
            },
            dataType: "json"
          });
          console.log("query: ", query);
          var templateResponse = Handlebars.compile( $("#message-response-template").html());
          var contextResponse = { 
            response: query,
            time: this.getCurrentTime()
          };          
          setTimeout(function() {
            this.$chatHistoryList.append(templateResponse(contextResponse));
            this.scrollToBottom();
          }.bind(this), 1500);
        }

        if(start_chat) {
          start_chat = false;
          var query = "";
          $.ajax({
            type: 'POST',
            async: false,
            url: "/request/query/",
            data: {"user_id": user_id, "img_name": img_path, "record_id": record_id},
            success: function(rdata){
              query = rdata["query"];
              record_id = rdata["record_id"];
            },
            dataType: "json"
          });
          var templateResponse = Handlebars.compile( $("#message-response-template").html());
          var contextResponse = { 
            response: query,
            time: this.getCurrentTime()
          };          
          setTimeout(function() {
            this.$chatHistoryList.append(templateResponse(contextResponse));
            this.scrollToBottom();
          }.bind(this), 1500);
        }
      },
      
      addMessage: function() {
        this.messageToSend = this.$textarea.val()
        this.render();
      },

      startMessage: function(){
        start_chat = true;
        this.render();
      },

      addMessageEnter: function(event) {
          // enter was pressed
          if (event.keyCode === 13) {
            this.addMessage();
          }
      },

      scrollToBottom: function() {
        this.$chatHistory.scrollTop(this.$chatHistory[0].scrollHeight);
      },

      getCurrentTime: function() {
        return new Date().toLocaleTimeString().
                replace(/([\d]+:[\d]{2})(:[\d]{2})(.*)/, "$1$3");
      },
    };

    chat.init();
  }
)();


function change_img(){
  $.ajax({
    type: 'GET',
    async: false,
    url: "/request/img",
    data: {"user_id": user_id},
    success: function(rdata){
      img_path = rdata["img_path"];
    },
    dataType: "json"
  });
  $(".guess-img").replaceWith(
    '<div class="guess-img"><img src="'+ img_path +'"/></div>'
  );
}
