module.exports = {
  plugins: [ 'mup-node' ],

  servers: {
    one: {
      // TODO: set host address, username, and authentication method
      host: '167.99.142.42',
      username: 'root',
      pem: '~/.ssh/id_rsa'
      // password: 'server-password'
      // or neither for authenticate from ssh-agent
    }
  },

  app: {
    // TODO: change app name and path
    name: 'app',
    type: 'node',
    nodeVersion: '6.9.1',
    path: '../',

    servers: {
      one: {},
    },

    //buildOptions: {
    //  serverOnly: true,
    //},

    env: {
      // TODO: Change to your app's url
      // If you are using ssl, it needs to start with https://
      ROOT_URL: 'http://app.com',
      MONGO_URL: 'mongodb://mongodb/meteor',
      MONGO_OPLOG_URL: 'mongodb://mongodb/local',
    },

    docker: {
      //image: 'zodern/meteor:root',
    },

    // Show progress bar while uploading bundle to server
    // You might need to disable it on CI servers
    //enableUploadProgressBar: true
  },

  /*mongo: {
    version: '3.4.1',
    servers: {
      one: {}
    }
  },*/

  // (Optional)
  // Use the proxy to setup ssl or to route requests to the correct
  // app when there are several apps

  // proxy: {
  //   domains: 'mywebsite.com,www.mywebsite.com',

  //   ssl: {
  //     // Enable Let's Encrypt
  //     letsEncryptEmail: 'email@domain.com'
  //   }
  // }
};
